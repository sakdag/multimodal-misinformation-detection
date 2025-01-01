import torch
import logging
import torch.nn.functional as F
from PIL import Image
from transformers import AutoTokenizer, AutoModel, Swinv2Model
from torchvision import transforms
from src.model.model import MisinformationDetectionModel

logger = logging.getLogger(__name__)


class MisinformationPredictor:
    def __init__(
        self,
        model_path,
        device="cuda" if torch.cuda.is_available() else "cpu",
        embed_dim=256,
        num_heads=8,
        dropout=0.1,
        hidden_dim=64,
        num_classes=3,
        mlp_ratio=4.0,
        text_input_dim=384,
        image_input_dim=1024,
        fused_attn=False,
        text_encoder="microsoft/deberta-v3-xsmall",
    ):
        """
        Initialize the predictor with a trained model and required encoders.

        Args:
            model_path: Path to the saved model checkpoint
            text_encoder: Name/path of the text encoder model
            device: Device to run inference on
            Other args: Model architecture parameters
        """
        self.device = torch.device(device)

        # Initialize tokenizer and encoders
        logger.info("Loading encoders...")
        self.tokenizer = AutoTokenizer.from_pretrained(text_encoder)
        self.text_encoder = AutoModel.from_pretrained(text_encoder).to(self.device)
        self.image_encoder = Swinv2Model.from_pretrained(
            "microsoft/swinv2-base-patch4-window8-256"
        ).to(self.device)

        # Set encoders to eval mode
        self.text_encoder.eval()
        self.image_encoder.eval()

        # Initialize model
        self.model = MisinformationDetectionModel(
            text_input_dim=text_input_dim,
            image_input_dim=image_input_dim,
            embed_dim=embed_dim,
            num_heads=num_heads,
            dropout=dropout,
            hidden_dim=hidden_dim,
            num_classes=num_classes,
            mlp_ratio=mlp_ratio,
            fused_attn=fused_attn,
        ).to(self.device)

        # Load model weights
        logger.info(f"Loading model from {model_path}")
        checkpoint = torch.load(model_path, map_location=self.device)
        self.model.load_state_dict(checkpoint["model_state_dict"])
        self.model.eval()

        # Image preprocessing
        self.image_transform = transforms.Compose(
            [
                transforms.Resize((256, 256)),
                transforms.ToTensor(),
                transforms.Normalize(
                    mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
                ),
            ]
        )

        # Class mapping
        self.idx_to_label = {0: "support", 1: "not_enough_information", 2: "refute"}

    def process_image(self, image_path):
        """Process image from path to tensor."""
        try:
            image = Image.open(image_path).convert("RGB")
            image = self.image_transform(image).unsqueeze(0)  # Add batch dimension
            return image.to(self.device)
        except Exception as e:
            logger.error(f"Error processing image {image_path}: {e}")
            return None

    @torch.no_grad()
    def evaluate(
        self, claim_text, claim_image_path, evidence_text, evidence_image_path
    ):
        """
        Evaluate a single claim-evidence pair.

        Args:
            claim_text (str): The claim text
            claim_image_path (str): Path to the claim image
            evidence_text (str): The evidence text
            evidence_image_path (str): Path to the evidence image

        Returns:
            dict: Dictionary containing predictions from all modality combinations
        """
        try:
            # Process text inputs
            claim_text_inputs = self.tokenizer(
                claim_text,
                truncation=True,
                padding="max_length",
                max_length=512,
                return_tensors="pt",
            ).to(self.device)

            evidence_text_inputs = self.tokenizer(
                evidence_text,
                truncation=True,
                padding="max_length",
                max_length=512,
                return_tensors="pt",
            ).to(self.device)

            # Get text embeddings
            claim_text_embeds = self.text_encoder(**claim_text_inputs).last_hidden_state
            evidence_text_embeds = self.text_encoder(
                **evidence_text_inputs
            ).last_hidden_state

            # Process image inputs
            claim_image = self.process_image(claim_image_path)
            evidence_image = self.process_image(evidence_image_path)

            # Process claim image
            if claim_image is not None:
                claim_image_embeds = self.image_encoder(claim_image).last_hidden_state
            else:
                logger.warning(
                    "Claim image processing failed, setting embedding to None"
                )
                claim_image_embeds = None

            # Process evidence image
            if evidence_image is not None:
                evidence_image_embeds = self.image_encoder(
                    evidence_image
                ).last_hidden_state
            else:
                logger.warning(
                    "Evidence image processing failed, setting embedding to None"
                )
                evidence_image_embeds = None

            # Get model predictions
            (y_t_t, y_t_i), (y_i_t, y_i_i) = self.model(
                X_t=claim_text_embeds,
                X_i=claim_image_embeds,
                E_t=evidence_text_embeds,
                E_i=evidence_image_embeds,
            )

            # Process predictions with confidence scores
            predictions = {}

            def process_output(output, path_name):
                if output is not None:
                    probs = F.softmax(output, dim=-1)
                    pred_idx = probs.argmax(dim=-1).item()
                    confidence = probs[0][pred_idx].item()
                    return {
                        "label": self.idx_to_label[pred_idx],
                        "confidence": confidence,
                        "probabilities": {
                            self.idx_to_label[i]: p.item()
                            for i, p in enumerate(probs[0])
                        },
                    }
                return None

            predictions["text_text"] = process_output(y_t_t, "text_text")
            predictions["text_image"] = process_output(y_t_i, "text_image")
            predictions["image_text"] = process_output(y_i_t, "image_text")
            predictions["image_image"] = process_output(y_i_i, "image_image")

            return {
                path: pred["label"] if pred else None
                for path, pred in predictions.items()
            }

        except Exception as e:
            logger.error(f"Error during evaluation: {e}")
            return None


if __name__ == "__main__":
    # Example usage
    logging.basicConfig(level=logging.INFO)

    predictor = MisinformationPredictor(model_path="ckpts/model.pt", device="cpu")

    # Example prediction
    predictions = predictor.evaluate(
        claim_text="Musician Kodak Black was shot outside of a nightclub in Florida in December 2016.",
        claim_image_path="./data/raw/factify/extracted/images/test/0_claim.jpg",
        evidence_text="On 26 December 2016, the web site Gummy Post published an article claiming \
                        that musician Kodak Black was shot outside a nightclub in Florida. \
                        This article is a hoax. While Gummy Post cited a 'police report', no records exist \
                        of any shooting involving Kodak Black (real name Dieuson Octave) in Florida during December 2016. \
                        Additionally, the video Gummy Post shared as evidence showed an unrelated crime scene.",
        evidence_image_path="./data/raw/factify/extracted/images/test/0_evidence.jpg",
    )

    print(predictions)
    # Print predictions
    # if predictions:
    #     print("\nPredictions:")
    #     for path, pred in predictions.items():
    #         if pred:
    #             print(f"\n{path}:")
    #             print(f"  Label: {pred['label']}")
    #             print(f"  Confidence: {pred['confidence']:.4f}")
    #             print("  Probabilities:")
    #             for label, prob in pred["probabilities"].items():
    #                 print(f"    {label}: {prob:.4f}")
