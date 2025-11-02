import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from moral_evaluation import load_model, CHARACTERS, CHAR_TO_IDX

def extract_attention_weights(model, scenario):
    """
    Extract attention weights from all layers and heads for a given scenario.

    Args:
        model: The MoralReasoningTransformer model
        scenario: Tensor of shape (batch, 2, 23) representing the scenario

    Returns:
        attention_by_layer: dict mapping layer_idx -> attention weights
                           Each attention weight has shape (batch, num_heads, seq_len, seq_len)
    """
    model.eval()
    attention_by_layer = {}

    with torch.no_grad():
        # Forward pass through embedding
        batch_size = scenario.shape[0]
        outcome_0 = scenario[:, 0, :]
        outcome_1 = scenario[:, 1, :]

        tokens_0 = model.encode_outcome(outcome_0, team_id=0)
        tokens_1 = model.encode_outcome(outcome_1, team_id=1)

        cls_tokens = model.cls_token.expand(batch_size, -1, -1)
        x = torch.cat([cls_tokens, tokens_0, tokens_1], dim=1)

        # Forward pass through transformer layers, extracting attention
        for layer_idx, layer in enumerate(model.transformer.layers):
            # Apply layer norm (norm_first=True)
            x_norm = layer.norm1(x)

            # Get attention weights
            attn_output, attn_weights = layer.self_attn(
                x_norm, x_norm, x_norm,
                need_weights=True,
                average_attn_weights=False  # Keep per-head weights
            )

            # Store attention weights
            attention_by_layer[layer_idx] = attn_weights

            # Continue forward pass
            x = x + layer.dropout1(attn_output)
            x = x + layer._ff_block(layer.norm2(x))

    return attention_by_layer


def aggregate_character_attention(attention_by_layer, team_id=0):
    """
    Aggregate attention weights across all layers and heads.
    Focus on character-to-character attention for a specific team.

    Args:
        attention_by_layer: dict mapping layer_idx -> attention weights
        team_id: Which team to analyze (0 or 1)

    Returns:
        aggregated_attention: numpy array of shape (23, 23) showing
                            how each character attends to each other character
    """
    num_layers = len(attention_by_layer)

    # Average across all layers and heads
    all_attention = []
    for layer_idx in range(num_layers):
        attn = attention_by_layer[layer_idx]  # (batch, num_heads, seq_len, seq_len)
        all_attention.append(attn)

    # Stack and average: (num_layers, batch, num_heads, seq_len, seq_len)
    stacked_attention = torch.stack(all_attention, dim=0)

    # Average across layers and heads: (batch, seq_len, seq_len)
    avg_attention = stacked_attention.mean(dim=[0, 2])

    # Convert to numpy and extract character-to-character attention
    avg_attention_np = avg_attention.cpu().numpy()[0]  # (seq_len, seq_len)

    # Extract attention for the specified team
    # Position 0: CLS token
    # Positions 1-23: Team 0 characters
    # Positions 24-46: Team 1 characters
    if team_id == 0:
        char_start, char_end = 1, 24
    else:
        char_start, char_end = 24, 47

    # Extract character-to-character attention submatrix
    char_attention = avg_attention_np[char_start:char_end, char_start:char_end]

    return char_attention


def extract_cls_attention(attention_by_layer, team_id=0):
    """
    Extract how the CLS token attends to each character.

    Args:
        attention_by_layer: dict mapping layer_idx -> attention weights
        team_id: Which team to analyze (0 or 1)

    Returns:
        cls_attention: numpy array of shape (23,) showing CLS attention to each character
    """
    num_layers = len(attention_by_layer)

    # Average across all layers and heads
    all_attention = []
    for layer_idx in range(num_layers):
        attn = attention_by_layer[layer_idx]  # (batch, num_heads, seq_len, seq_len)
        all_attention.append(attn)

    # Stack and average: (num_layers, batch, num_heads, seq_len, seq_len)
    stacked_attention = torch.stack(all_attention, dim=0)

    # Average across layers and heads: (batch, seq_len, seq_len)
    avg_attention = stacked_attention.mean(dim=[0, 2])

    # Convert to numpy
    avg_attention_np = avg_attention.cpu().numpy()[0]  # (seq_len, seq_len)

    # Extract CLS token attention (row 0) to characters
    if team_id == 0:
        char_start, char_end = 1, 24
    else:
        char_start, char_end = 24, 47

    cls_to_chars = avg_attention_np[0, char_start:char_end]

    return cls_to_chars


def plot_aggregated_attention(char_attention, characters, output_path='character_attention_aggregated.png'):
    """
    Plot the aggregated character-to-character attention matrix.

    Args:
        char_attention: numpy array of shape (23, 23)
        characters: list of character names
        output_path: path to save the figure
    """
    plt.figure(figsize=(12, 10))

    sns.heatmap(
        char_attention,
        xticklabels=characters,
        yticklabels=characters,
        cmap='YlOrRd',
        annot=False,
        square=True,
        linewidths=0.5,
        cbar_kws={'label': 'Attention Weight'},
        vmin=0,
        vmax=char_attention.max()
    )

    plt.title('Aggregated Character-to-Character Attention\n(Averaged across all layers and heads)',
              fontsize=14, pad=20)
    plt.xlabel('Attended To', fontsize=12)
    plt.ylabel('Attending From', fontsize=12)
    plt.xticks(rotation=45, ha='right')
    plt.yticks(rotation=0)
    plt.tight_layout()

    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"Saved aggregated attention matrix to {output_path}")
    plt.close()


def plot_cls_attention(cls_attention, characters, output_path='character_attention_cls.png'):
    """
    Plot how the CLS token attends to each character.

    Args:
        cls_attention: numpy array of shape (23,)
        characters: list of character names
        output_path: path to save the figure
    """
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 10))

    # Bar plot
    x_pos = np.arange(len(characters))
    colors = plt.cm.YlOrRd(cls_attention / cls_attention.max())

    ax1.bar(x_pos, cls_attention, color=colors, edgecolor='black', linewidth=0.5)
    ax1.set_xticks(x_pos)
    ax1.set_xticklabels(characters, rotation=45, ha='right')
    ax1.set_ylabel('Attention Weight', fontsize=12)
    ax1.set_title('CLS Token Attention to Characters\n(Bar Plot)', fontsize=14, pad=20)
    ax1.grid(axis='y', alpha=0.3)

    # Heatmap
    cls_attention_2d = cls_attention.reshape(1, -1)
    sns.heatmap(
        cls_attention_2d,
        xticklabels=characters,
        yticklabels=['CLS Token'],
        cmap='YlOrRd',
        annot=False,
        cbar_kws={'label': 'Attention Weight'},
        vmin=0,
        vmax=cls_attention.max(),
        ax=ax2
    )
    ax2.set_title('CLS Token Attention to Characters\n(Heatmap)', fontsize=14, pad=20)
    ax2.set_xlabel('Character', fontsize=12)
    plt.setp(ax2.get_xticklabels(), rotation=45, ha='right')

    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"Saved CLS attention visualization to {output_path}")
    plt.close()


def create_all_characters_scenario(device):
    """
    Create a scenario where all characters are present (1 of each).

    Returns:
        scenario: Tensor of shape (1, 2, 23)
    """
    # Create vectors with 1 count for each character
    vec = np.ones(23, dtype=np.int64)

    # Create scenario with same characters on both teams
    scenario = torch.tensor([[vec, vec]], dtype=torch.long).to(device)

    return scenario


def main():
    print("Loading model...")
    model, device = load_model('best_model.pt')

    print("Creating scenario with all characters...")
    scenario = create_all_characters_scenario(device)

    print("Extracting attention weights...")
    attention_by_layer = extract_attention_weights(model, scenario)

    print(f"Extracted attention from {len(attention_by_layer)} layers")
    for layer_idx, attn in attention_by_layer.items():
        print(f"  Layer {layer_idx}: shape {attn.shape}")

    # Analyze team 0 (can also do team 1)
    print("\nAggregating character-to-character attention (Team 0)...")
    char_attention = aggregate_character_attention(attention_by_layer, team_id=0)
    print(f"Character attention matrix shape: {char_attention.shape}")
    print(f"Attention range: [{char_attention.min():.4f}, {char_attention.max():.4f}]")

    print("\nExtracting CLS token attention (Team 0)...")
    cls_attention = extract_cls_attention(attention_by_layer, team_id=0)
    print(f"CLS attention shape: {cls_attention.shape}")
    print(f"CLS attention range: [{cls_attention.min():.4f}, {cls_attention.max():.4f}]")

    print("\nCreating visualizations...")
    plot_aggregated_attention(char_attention, CHARACTERS, 'character_attention_aggregated.png')
    plot_cls_attention(cls_attention, CHARACTERS, 'character_attention_cls.png')

    print("\n" + "="*60)
    print("Experiment complete!")
    print("="*60)
    print("\nGenerated files:")
    print("  - character_attention_aggregated.png")
    print("  - character_attention_cls.png")

    # Print top 5 characters by CLS attention
    print("\nTop 5 characters by CLS attention:")
    top_indices = np.argsort(cls_attention)[::-1][:5]
    for i, idx in enumerate(top_indices, 1):
        print(f"  {i}. {CHARACTERS[idx]}: {cls_attention[idx]:.4f}")


if __name__ == '__main__':
    main()
