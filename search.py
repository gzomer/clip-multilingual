import torch

from .models import create_tokenizer, create_and_load_from_hub

class MultiLingualSearch:
    def __init__(self, images_embeddings, images_data, model = None, device='cpu'):
        self.model = model if model else create_and_load_from_hub()
        self.tokenizer = create_tokenizer()
        self.images_embeddings = images_embeddings
        self.images_data = images_data
        self.device = device

    def compare_embeddings(self, logit_scale, img_embs, txt_embs):
        # normalized features
        image_features = img_embs / img_embs.norm(dim=-1, keepdim=True)
        text_features = txt_embs / txt_embs.norm(dim=-1, keepdim=True)

        # cosine similarity as logits
        logits_per_image = logit_scale * image_features @ text_features.t()
        logits_per_text = logit_scale * text_features @ image_features.t()

        return logits_per_image, logits_per_text

    def compare_text_images(self, model, text, images_embeddings):
        tokens = self.tokenizer(text)
        tokens = {k: v.to(self.device) for k, v in tokens.items()}

        with torch.no_grad():
            txt_embs = model.caption_encoder(tokens)

        images_tensors = torch.tensor(images_embeddings)

        logit_scale = model.clip_model.logit_scale.exp().float().to('cpu')
        logits_images, logits_text = self.compare_embeddings(logit_scale, images_tensors.to('cpu'), txt_embs.to('cpu'))
        return logits_images.softmax(dim=0).cpu().detach().numpy()

    def search(self, text, amount=10):
        probs = self.compare_text_images(self.model, text, self.images_embeddings)
        images_probs = list(zip(self.images_data, [item[0] for item in probs.tolist()]))
        sorted_images = sorted(images_probs, key=lambda x:x[1], reverse=True)
        return [{'image': item[0], 'prob': item[1]} for item in sorted_images[:amount]]