import torch
import torch.nn as nn
import numpy as np

CHARACTERS = [
    'Intervention', 'Barrier', 'CrossingSignal',
    'Man', 'Woman', 'Pregnant', 'Stroller', 'OldMan', 'OldWoman',
    'Boy', 'Girl', 'Homeless', 'LargeWoman', 'LargeMan', 'Criminal',
    'MaleExecutive', 'FemaleExecutive', 'FemaleAthlete', 'MaleAthlete',
    'FemaleDoctor', 'MaleDoctor', 'Dog', 'Cat'
]

CHAR_TO_IDX = {char: idx for idx, char in enumerate(CHARACTERS)}

class MoralReasoningTransformer(nn.Module):
    def __init__(
        self,
        num_characters: int = 23,
        max_cardinality: int = 10,
        num_teams: int = 2,
        embed_dim: int = 64,
        num_heads: int = 2,
        num_layers: int = 2,
        dropout: float = 0.1
    ):
        super().__init__()

        self.num_characters = num_characters
        self.embed_dim = embed_dim
        self.char_embed_dim = embed_dim // 2
        self.card_team_embed_dim = embed_dim // 4

        # Compositional embeddings (character gets half, cardinality and team get quarter each)
        self.character_embedding = nn.Embedding(num_characters, self.char_embed_dim)
        self.cardinality_embedding = nn.Embedding(max_cardinality + 1, self.card_team_embed_dim)
        self.team_embedding = nn.Embedding(num_teams, self.card_team_embed_dim)

        # Transformer encoder
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=embed_dim,
            nhead=num_heads,
            dim_feedforward=embed_dim * 4,
            dropout=dropout,
            batch_first=True,
            norm_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

        # CLS token for aggregation
        self.cls_token = nn.Parameter(torch.randn(1, 1, embed_dim))

        # Classification head on CLS token
        self.classifier = nn.Sequential(
            nn.LayerNorm(embed_dim),
            nn.Linear(embed_dim, embed_dim // 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(embed_dim // 2, 1)
        )

    def encode_outcome(self, counts, team_id):
        batch_size = counts.shape[0]

        character_ids = torch.arange(
            self.num_characters,
            device=counts.device
        ).unsqueeze(0).expand(batch_size, -1)

        char_emb = self.character_embedding(character_ids)  # (batch, 23, embed_dim//2)
        card_emb = self.cardinality_embedding(counts)        # (batch, 23, embed_dim//4)

        team_id_tensor = torch.full(
            (batch_size, self.num_characters),
            team_id,
            device=counts.device,
            dtype=torch.long
        )
        team_emb = self.team_embedding(team_id_tensor)       # (batch, 23, embed_dim//4)

        # Concatenate along last dimension: embed_dim//2 + embed_dim//4 + embed_dim//4 = embed_dim
        tokens = torch.cat([char_emb, card_emb, team_emb], dim=-1)

        return tokens

    def forward(self, scenarios):
        batch_size = scenarios.shape[0]

        outcome_0 = scenarios[:, 0, :]
        outcome_1 = scenarios[:, 1, :]

        tokens_0 = self.encode_outcome(outcome_0, team_id=0)
        tokens_1 = self.encode_outcome(outcome_1, team_id=1)

        cls_tokens = self.cls_token.expand(batch_size, -1, -1)
        all_tokens = torch.cat([cls_tokens, tokens_0, tokens_1], dim=1)

        encoded = self.transformer(all_tokens)
        cls_output = encoded[:, 0, :]
        logits = self.classifier(cls_output)

        return logits


def load_model(path='best_model.pt'):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = MoralReasoningTransformer()
    model.load_state_dict(torch.load(path, map_location=device)['model_state_dict'])
    model.to(device).eval()
    return model, device


def get_probs(model, device, scenario_tuple):
    """
    scenario_tuple: (dict_0, dict_1) where dicts have character names as keys
    Returns: [prob_outcome_0, prob_outcome_1] that sum to 1.0
    """
    outcome_0, outcome_1 = scenario_tuple

    vec_0 = np.zeros(23, dtype=np.int64)
    vec_1 = np.zeros(23, dtype=np.int64)

    for char, count in outcome_0.items():
        vec_0[CHAR_TO_IDX[char]] = count
    for char, count in outcome_1.items():
        vec_1[CHAR_TO_IDX[char]] = count

    scenario_1 = torch.tensor([[vec_0, vec_1]], dtype=torch.long).to(device)
    scenario_2 = torch.tensor([[vec_1, vec_0]], dtype=torch.long).to(device)

    with torch.no_grad():
        probs_1 = torch.sigmoid(model(scenario_1)).item()
        probs_2 = torch.sigmoid(model(scenario_2)).item()
        prob1 = (probs_1 + (1 - probs_2))/2
        prob0 = 1 - prob1
        return [prob0, prob1]


# Usage:
# model, device = load_model('best_model.pt')
# prob = get_probs(model, device, ({'Man': 3}, {'Criminal': 3}))
# print(prob)
# [0.8888750448822975, 0.11112495511770248]