import torch
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from collections import defaultdict
from moral_evaluation import load_model, CHARACTERS, CHAR_TO_IDX


def get_attention_weights_for_scenario(model, device, scenario_dict_0, scenario_dict_1):
    """
    Extract attention weights for a specific scenario.

    Args:
        model: The trained model
        device: torch device
        scenario_dict_0: Dictionary for outcome 0 (e.g., {'Man': 1, 'Woman': 2})
        scenario_dict_1: Dictionary for outcome 1

    Returns:
        Dictionary mapping layer_idx -> attention weights (shape: batch, num_heads, seq_len, seq_len)
    """
    # Convert scenario to tensor format
    vec_0 = np.zeros(23, dtype=np.int64)
    vec_1 = np.zeros(23, dtype=np.int64)

    for char, count in scenario_dict_0.items():
        vec_0[CHAR_TO_IDX[char]] = count
    for char, count in scenario_dict_1.items():
        vec_1[CHAR_TO_IDX[char]] = count

    scenario_tensor = torch.tensor([[vec_0, vec_1]], dtype=torch.long).to(device)

    attention_by_layer = {}

    with torch.no_grad():
        batch_size = scenario_tensor.shape[0]

        # Encode scenarios
        outcome_0 = scenario_tensor[:, 0, :]
        outcome_1 = scenario_tensor[:, 1, :]

        tokens_0 = model.encode_outcome(outcome_0, team_id=0)
        tokens_1 = model.encode_outcome(outcome_1, team_id=1)

        cls_tokens = model.cls_token.expand(batch_size, -1, -1)
        all_tokens = torch.cat([cls_tokens, tokens_0, tokens_1], dim=1)

        # Process through each layer to extract attention
        x = all_tokens
        for layer_idx, layer in enumerate(model.transformer.layers):
            # Apply layer normalization
            x_norm = layer.norm1(x)

            # Get attention with weights
            attn_output, attn_weights = layer.self_attn(
                x_norm, x_norm, x_norm,
                need_weights=True,
                average_attn_weights=False  # Keep per-head weights
            )

            attention_by_layer[layer_idx] = attn_weights

            # Continue forward pass
            x = x + layer.dropout1(attn_output)
            x = x + layer._ff_block(layer.norm2(x))

    return attention_by_layer


def generate_diverse_scenarios(num_scenarios=100):
    """
    Generate diverse scenarios for computing average attention patterns.

    Returns:
        List of (scenario_dict_0, scenario_dict_1) tuples
    """
    scenarios = []
    sentient_chars = [c for c in CHARACTERS if c not in ['Intervention', 'Barrier', 'CrossingSignal']]

    # Random scenarios
    for _ in range(num_scenarios):
        num_chars_0 = np.random.randint(1, 5)
        num_chars_1 = np.random.randint(1, 5)

        chars_0 = np.random.choice(sentient_chars, size=num_chars_0, replace=True)
        chars_1 = np.random.choice(sentient_chars, size=num_chars_1, replace=True)

        scenario_0 = {}
        scenario_1 = {}

        for char in chars_0:
            scenario_0[char] = scenario_0.get(char, 0) + 1
        for char in chars_1:
            scenario_1[char] = scenario_1.get(char, 0) + 1

        scenarios.append((scenario_0, scenario_1))

    return scenarios


def compute_character_attention_matrix(model, device, scenarios):
    """
    Compute average attention matrix for characters across multiple scenarios.

    Returns:
        Dictionary with structure:
        {
            layer_idx: {
                head_idx: {
                    'team_0': attention_matrix (23 x 23),
                    'team_1': attention_matrix (23 x 23),
                    'cls_to_team_0': attention_vector (23,),
                    'cls_to_team_1': attention_vector (23,)
                }
            }
        }
    """
    num_layers = len(model.transformer.layers)
    num_heads = model.transformer.layers[0].self_attn.num_heads
    num_chars = 23

    # Initialize accumulators
    attention_accum = defaultdict(lambda: defaultdict(lambda: {
        'team_0': np.zeros((num_chars, num_chars)),
        'team_1': np.zeros((num_chars, num_chars)),
        'cls_to_team_0': np.zeros(num_chars),
        'cls_to_team_1': np.zeros(num_chars),
        'count': 0
    }))

    print(f"Processing {len(scenarios)} scenarios...")

    for idx, (scenario_0, scenario_1) in enumerate(scenarios):
        if idx % 20 == 0:
            print(f"  Progress: {idx}/{len(scenarios)}")

        attention_by_layer = get_attention_weights_for_scenario(
            model, device, scenario_0, scenario_1
        )

        for layer_idx, attn_weights in attention_by_layer.items():
            # attn_weights shape: (batch, num_heads, seq_len, seq_len)
            # seq_len = 1 (CLS) + 23 (team 0) + 23 (team 1) = 47

            attn = attn_weights[0].cpu().numpy()  # (num_heads, 47, 47)

            for head_idx in range(num_heads):
                head_attn = attn[head_idx]  # (47, 47)

                # Extract regions:
                # CLS: position 0
                # Team 0 characters: positions 1-23
                # Team 1 characters: positions 24-46

                # Team 0 to Team 0 attention
                team_0_attn = head_attn[1:24, 1:24]  # (23, 23)
                attention_accum[layer_idx][head_idx]['team_0'] += team_0_attn

                # Team 1 to Team 1 attention
                team_1_attn = head_attn[24:47, 24:47]  # (23, 23)
                attention_accum[layer_idx][head_idx]['team_1'] += team_1_attn

                # CLS attention to Team 0
                cls_to_team_0 = head_attn[0, 1:24]  # (23,)
                attention_accum[layer_idx][head_idx]['cls_to_team_0'] += cls_to_team_0

                # CLS attention to Team 1
                cls_to_team_1 = head_attn[0, 24:47]  # (23,)
                attention_accum[layer_idx][head_idx]['cls_to_team_1'] += cls_to_team_1

                attention_accum[layer_idx][head_idx]['count'] += 1

    # Average the accumulated attention
    attention_matrices = {}
    for layer_idx in attention_accum:
        attention_matrices[layer_idx] = {}
        for head_idx in attention_accum[layer_idx]:
            count = attention_accum[layer_idx][head_idx]['count']
            attention_matrices[layer_idx][head_idx] = {
                'team_0': attention_accum[layer_idx][head_idx]['team_0'] / count,
                'team_1': attention_accum[layer_idx][head_idx]['team_1'] / count,
                'cls_to_team_0': attention_accum[layer_idx][head_idx]['cls_to_team_0'] / count,
                'cls_to_team_1': attention_accum[layer_idx][head_idx]['cls_to_team_1'] / count,
            }

    return attention_matrices


def plot_attention_heatmap(attention_matrix, characters, title, save_path):
    """Plot a single attention heatmap."""
    # Filter to sentient characters for cleaner visualization
    sentient_chars = [c for c in characters if c not in ['Intervention', 'Barrier', 'CrossingSignal']]
    sentient_indices = [i for i, c in enumerate(characters) if c in sentient_chars]

    filtered_attn = attention_matrix[np.ix_(sentient_indices, sentient_indices)]

    fig, ax = plt.subplots(figsize=(14, 12))

    sns.heatmap(filtered_attn,
                xticklabels=sentient_chars,
                yticklabels=sentient_chars,
                cmap='YlOrRd',
                annot=False,
                fmt='.3f',
                square=True,
                linewidths=0.5,
                cbar_kws={'label': 'Attention Weight'},
                vmin=0,
                vmax=None,
                ax=ax)

    ax.set_title(title, fontsize=16, fontweight='bold', pad=20)
    ax.set_xlabel('Attending To (Character)', fontsize=12, fontweight='bold')
    ax.set_ylabel('Attending From (Character)', fontsize=12, fontweight='bold')
    plt.xticks(rotation=45, ha='right')
    plt.yticks(rotation=0)

    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"Saved: {save_path}")

    return fig


def plot_layer_head_attention_matrices(attention_matrices, save_path_prefix='attention_matrix'):
    """Plot attention matrices for all layer-head combinations."""

    for layer_idx in sorted(attention_matrices.keys()):
        for head_idx in sorted(attention_matrices[layer_idx].keys()):
            # Team 0 attention
            team_0_attn = attention_matrices[layer_idx][head_idx]['team_0']
            plot_attention_heatmap(
                team_0_attn,
                CHARACTERS,
                f'Layer {layer_idx}, Head {head_idx} - Team 0 Self-Attention',
                f'{save_path_prefix}_L{layer_idx}_H{head_idx}_team0.png'
            )

            # Team 1 attention
            team_1_attn = attention_matrices[layer_idx][head_idx]['team_1']
            plot_attention_heatmap(
                team_1_attn,
                CHARACTERS,
                f'Layer {layer_idx}, Head {head_idx} - Team 1 Self-Attention',
                f'{save_path_prefix}_L{layer_idx}_H{head_idx}_team1.png'
            )


def plot_cls_attention(attention_matrices, save_path='attention_matrix_cls.png'):
    """Plot CLS token attention across all layers and heads."""
    num_layers = len(attention_matrices)
    num_heads = len(attention_matrices[0])

    sentient_chars = [c for c in CHARACTERS if c not in ['Intervention', 'Barrier', 'CrossingSignal']]
    sentient_indices = [i for i, c in enumerate(CHARACTERS) if c in sentient_chars]

    fig, axes = plt.subplots(num_layers, num_heads * 2, figsize=(20, 8))

    for layer_idx in range(num_layers):
        for head_idx in range(num_heads):
            # Team 0 CLS attention
            ax_team0 = axes[layer_idx, head_idx * 2]
            cls_attn_0 = attention_matrices[layer_idx][head_idx]['cls_to_team_0'][sentient_indices]

            ax_team0.bar(range(len(sentient_chars)), cls_attn_0, alpha=0.7, color='skyblue')
            ax_team0.set_title(f'L{layer_idx}H{head_idx} Team0', fontsize=10)
            ax_team0.set_xticks(range(len(sentient_chars)))
            ax_team0.set_xticklabels(sentient_chars, rotation=90, fontsize=8)
            ax_team0.set_ylabel('Attention', fontsize=9)
            ax_team0.grid(axis='y', alpha=0.3)

            # Team 1 CLS attention
            ax_team1 = axes[layer_idx, head_idx * 2 + 1]
            cls_attn_1 = attention_matrices[layer_idx][head_idx]['cls_to_team_1'][sentient_indices]

            ax_team1.bar(range(len(sentient_chars)), cls_attn_1, alpha=0.7, color='salmon')
            ax_team1.set_title(f'L{layer_idx}H{head_idx} Team1', fontsize=10)
            ax_team1.set_xticks(range(len(sentient_chars)))
            ax_team1.set_xticklabels(sentient_chars, rotation=90, fontsize=8)
            ax_team1.set_ylabel('Attention', fontsize=9)
            ax_team1.grid(axis='y', alpha=0.3)

    plt.suptitle('CLS Token Attention to Characters (Averaged Across Scenarios)',
                 fontsize=16, fontweight='bold', y=1.00)
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"Saved: {save_path}")

    return fig


def plot_aggregated_attention(attention_matrices, save_path='attention_matrix_aggregated.png'):
    """Plot aggregated attention across all layers and heads."""

    # Aggregate Team 0 attention
    team_0_agg = np.zeros((23, 23))
    team_1_agg = np.zeros((23, 23))
    count = 0

    for layer_idx in attention_matrices:
        for head_idx in attention_matrices[layer_idx]:
            team_0_agg += attention_matrices[layer_idx][head_idx]['team_0']
            team_1_agg += attention_matrices[layer_idx][head_idx]['team_1']
            count += 1

    team_0_agg /= count
    team_1_agg /= count

    # Plot both teams
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(24, 10))

    sentient_chars = [c for c in CHARACTERS if c not in ['Intervention', 'Barrier', 'CrossingSignal']]
    sentient_indices = [i for i, c in enumerate(CHARACTERS) if c in sentient_chars]

    # Team 0
    filtered_team_0 = team_0_agg[np.ix_(sentient_indices, sentient_indices)]
    sns.heatmap(filtered_team_0,
                xticklabels=sentient_chars,
                yticklabels=sentient_chars,
                cmap='YlOrRd',
                annot=False,
                square=True,
                linewidths=0.5,
                cbar_kws={'label': 'Avg Attention'},
                vmin=0,
                ax=ax1)
    ax1.set_title('Team 0 - Aggregated Attention (All Layers & Heads)', fontsize=14, fontweight='bold')
    ax1.set_xlabel('Attending To', fontsize=11, fontweight='bold')
    ax1.set_ylabel('Attending From', fontsize=11, fontweight='bold')
    plt.sca(ax1)
    plt.xticks(rotation=45, ha='right')
    plt.yticks(rotation=0)

    # Team 1
    filtered_team_1 = team_1_agg[np.ix_(sentient_indices, sentient_indices)]
    sns.heatmap(filtered_team_1,
                xticklabels=sentient_chars,
                yticklabels=sentient_chars,
                cmap='YlOrRd',
                annot=False,
                square=True,
                linewidths=0.5,
                cbar_kws={'label': 'Avg Attention'},
                vmin=0,
                ax=ax2)
    ax2.set_title('Team 1 - Aggregated Attention (All Layers & Heads)', fontsize=14, fontweight='bold')
    ax2.set_xlabel('Attending To', fontsize=11, fontweight='bold')
    ax2.set_ylabel('Attending From', fontsize=11, fontweight='bold')
    plt.sca(ax2)
    plt.xticks(rotation=45, ha='right')
    plt.yticks(rotation=0)

    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"Saved: {save_path}")

    return fig


def plot_character_attention_ranking(attention_matrices, save_path='attention_matrix_ranking.png'):
    """Plot which characters receive the most/least attention overall."""

    sentient_chars = [c for c in CHARACTERS if c not in ['Intervention', 'Barrier', 'CrossingSignal']]
    sentient_indices = [i for i, c in enumerate(CHARACTERS) if c in sentient_chars]

    # Compute average attention received by each character
    attention_received = np.zeros(len(sentient_chars))
    count = 0

    for layer_idx in attention_matrices:
        for head_idx in attention_matrices[layer_idx]:
            # Average across both teams
            team_0 = attention_matrices[layer_idx][head_idx]['team_0']
            team_1 = attention_matrices[layer_idx][head_idx]['team_1']

            # Sum attention received (column sums)
            for i, idx in enumerate(sentient_indices):
                attention_received[i] += team_0[:, idx].mean() + team_1[:, idx].mean()

            count += 2  # counting both teams

    attention_received /= count

    # Sort by attention
    sorted_indices = np.argsort(attention_received)[::-1]
    sorted_chars = [sentient_chars[i] for i in sorted_indices]
    sorted_attn = attention_received[sorted_indices]

    # Plot
    fig, ax = plt.subplots(figsize=(14, 8))

    colors = plt.cm.RdYlGn(np.linspace(0.3, 0.9, len(sorted_chars)))
    bars = ax.barh(range(len(sorted_chars)), sorted_attn, color=colors, alpha=0.8, edgecolor='black')

    ax.set_yticks(range(len(sorted_chars)))
    ax.set_yticklabels(sorted_chars)
    ax.set_xlabel('Average Attention Received', fontsize=12, fontweight='bold')
    ax.set_title('Character Attention Ranking\n(Averaged Across All Layers, Heads, and Scenarios)',
                 fontsize=14, fontweight='bold')
    ax.grid(axis='x', alpha=0.3)
    ax.invert_yaxis()

    # Add value labels
    for i, (bar, val) in enumerate(zip(bars, sorted_attn)):
        ax.text(val, bar.get_y() + bar.get_height()/2,
                f' {val:.4f}',
                va='center', fontsize=9)

    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"Saved: {save_path}")

    return fig


def main():
    print("Loading model...")
    model, device = load_model('best_model.pt')

    print("\n" + "="*70)
    print("GENERATING SCENARIOS")
    print("="*70)
    scenarios = generate_diverse_scenarios(num_scenarios=100)
    print(f"Generated {len(scenarios)} scenarios")

    print("\n" + "="*70)
    print("COMPUTING ATTENTION MATRICES")
    print("="*70)
    attention_matrices = compute_character_attention_matrix(model, device, scenarios)

    print("\n" + "="*70)
    print("GENERATING VISUALIZATIONS")
    print("="*70)

    # 1. Individual layer-head attention matrices (8 plots: 2 layers × 2 heads × 2 teams)
    print("\n1. Layer-Head Attention Matrices...")
    plot_layer_head_attention_matrices(attention_matrices)

    # 2. CLS token attention patterns
    print("\n2. CLS Token Attention...")
    plot_cls_attention(attention_matrices)

    # 3. Aggregated attention across all layers/heads
    print("\n3. Aggregated Attention...")
    plot_aggregated_attention(attention_matrices)

    # 4. Character attention ranking
    print("\n4. Character Attention Ranking...")
    plot_character_attention_ranking(attention_matrices)

    print("\n" + "="*70)
    print("ANALYSIS COMPLETE")
    print("="*70)
    print("\nGenerated files:")
    print("  - attention_matrix_L{0,1}_H{0,1}_team{0,1}.png: Individual layer-head attention (8 files)")
    print("  - attention_matrix_cls.png: CLS token attention patterns")
    print("  - attention_matrix_aggregated.png: Aggregated attention across layers/heads")
    print("  - attention_matrix_ranking.png: Character attention ranking")


if __name__ == "__main__":
    main()
