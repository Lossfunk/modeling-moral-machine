import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from collections import defaultdict
from moral_evaluation import load_model, get_probs, CHAR_TO_IDX, CHARACTERS
import bias_analysis


class AttentionHookManager:
    """Manages attention hooks for transformer layers."""

    def __init__(self, model):
        self.model = model
        self.attention_weights = {}
        self.hooks = []

    def register_hooks(self):
        """Register hooks on all transformer encoder layers."""
        for layer_idx, layer in enumerate(self.model.transformer.layers):
            # Hook into the self-attention module
            hook = layer.self_attn.register_forward_hook(
                self._create_hook_fn(layer_idx)
            )
            self.hooks.append(hook)

    def _create_hook_fn(self, layer_idx):
        """Create a hook function for a specific layer."""
        def hook_fn(module, input, output):
            # For MultiheadAttention, output is (attn_output, attn_weights)
            # when return_attn is False, we need to compute manually
            # We'll access the attention weights through the saved tensors

            # Store attention weights if available
            # Note: PyTorch MultiheadAttention doesn't return weights by default
            # We need to modify our approach to capture them
            pass

        return hook_fn

    def remove_hooks(self):
        """Remove all registered hooks."""
        for hook in self.hooks:
            hook.remove()
        self.hooks = []
        self.attention_weights = {}


class AttentionAnalyzer:
    """Analyzes attention patterns for bias detection."""

    def __init__(self, model, device):
        self.model = model
        self.device = device
        self.num_layers = len(model.transformer.layers)
        self.num_heads = model.transformer.layers[0].self_attn.num_heads

    def get_attention_weights(self, scenario_tensor):
        """
        Extract attention weights from all layers and heads.
        Modified to manually compute attention.
        """
        attention_by_layer = {}

        # We need to modify the model temporarily to capture attention
        # Since PyTorch's built-in MultiheadAttention doesn't expose weights easily,
        # we'll do a forward pass and manually extract from each layer

        with torch.no_grad():
            batch_size = scenario_tensor.shape[0]

            # Encode scenarios (copied from model's forward)
            outcome_0 = scenario_tensor[:, 0, :]
            outcome_1 = scenario_tensor[:, 1, :]

            tokens_0 = self.model.encode_outcome(outcome_0, team_id=0)
            tokens_1 = self.model.encode_outcome(outcome_1, team_id=1)

            cls_tokens = self.model.cls_token.expand(batch_size, -1, -1)
            all_tokens = torch.cat([cls_tokens, tokens_0, tokens_1], dim=1)

            # Process through each layer manually to extract attention
            x = all_tokens
            for layer_idx, layer in enumerate(self.model.transformer.layers):
                # Apply layer normalization if norm_first
                x_norm = layer.norm1(x)

                # Get attention output with weights
                # We'll use the internal _sa_block which calls self_attn
                attn_output, attn_weights = layer.self_attn(
                    x_norm, x_norm, x_norm,
                    need_weights=True,
                    average_attn_weights=False  # Keep per-head weights
                )

                attention_by_layer[layer_idx] = attn_weights

                # Continue the forward pass
                x = x + layer.dropout1(attn_output)
                x = x + layer._ff_block(layer.norm2(x))

        return attention_by_layer

    def compute_attention_for_scenario(self, scenario_tuple):
        """
        Compute attention weights for a specific scenario.

        Args:
            scenario_tuple: (dict_0, dict_1) as in bias_analysis

        Returns:
            Dictionary with attention weights by layer
        """
        outcome_0, outcome_1 = scenario_tuple

        vec_0 = np.zeros(23, dtype=np.int64)
        vec_1 = np.zeros(23, dtype=np.int64)

        for char, count in outcome_0.items():
            vec_0[CHAR_TO_IDX[char]] = count
        for char, count in outcome_1.items():
            vec_1[CHAR_TO_IDX[char]] = count

        scenario_tensor = torch.tensor([[vec_0, vec_1]], dtype=torch.long).to(self.device)

        return self.get_attention_weights(scenario_tensor)

    def analyze_bias_by_layer_head(self, bias_fn, bias_name):
        """
        Analyze how bias emerges across layers and heads.

        Args:
            bias_fn: Function that returns list of scenario tuples
            bias_name: Name of the bias being analyzed

        Returns:
            Dictionary with layer-head analysis
        """
        print(f"\nAnalyzing {bias_name} across layers and heads...")

        # Get scenarios from bias function
        scenarios = bias_fn()

        layer_head_contributions = defaultdict(lambda: defaultdict(list))

        for scenario in scenarios:
            # Get model prediction
            probs = get_probs(self.model, self.device, scenario)
            bias_score = probs[0] - 0.5

            # Get attention weights
            attention_by_layer = self.compute_attention_for_scenario(scenario)

            # Analyze each layer and head
            for layer_idx, attn_weights in attention_by_layer.items():
                # attn_weights shape: (batch, num_heads, seq_len, seq_len)
                # Focus on CLS token attention (position 0)
                cls_attention = attn_weights[0, :, 0, :]  # (num_heads, seq_len)

                for head_idx in range(cls_attention.shape[0]):
                    head_attn = cls_attention[head_idx].cpu().numpy()

                    # Store attention pattern with bias score
                    layer_head_contributions[layer_idx][head_idx].append({
                        'bias_score': bias_score,
                        'attention': head_attn,
                        'scenario': scenario
                    })

        return layer_head_contributions

    def compute_layer_head_importance(self, layer_head_contributions):
        """
        Compute which layers/heads are most important for bias.

        Uses correlation between attention patterns and bias scores.
        """
        importance = {}

        for layer_idx, heads in layer_head_contributions.items():
            importance[layer_idx] = {}

            for head_idx, samples in heads.items():
                if len(samples) == 0:
                    continue

                # Extract bias scores and average attention weights
                bias_scores = np.array([s['bias_score'] for s in samples])
                attention_patterns = np.array([s['attention'] for s in samples])

                # Compute variance in attention (more variance = more selective)
                attention_variance = np.mean(np.var(attention_patterns, axis=1))

                # Compute correlation with bias
                # Use attention on specific character positions
                correlations = []
                for pos in range(attention_patterns.shape[1]):
                    if np.std(attention_patterns[:, pos]) > 0:
                        corr = np.corrcoef(bias_scores, attention_patterns[:, pos])[0, 1]
                        if not np.isnan(corr):
                            correlations.append(abs(corr))

                avg_correlation = np.mean(correlations) if correlations else 0

                importance[layer_idx][head_idx] = {
                    'attention_variance': attention_variance,
                    'avg_correlation': avg_correlation,
                    'importance_score': attention_variance * avg_correlation
                }

        return importance


# Functions to generate scenarios for each bias type
def get_legality_scenarios():
    """Generate scenarios for legality bias analysis."""
    character_types = ['Man', 'Woman', 'Boy', 'Girl', 'OldMan', 'OldWoman']
    scenarios = []
    for char in character_types:
        scenarios.append(({'CrossingSignal': 1, char: 1}, {'CrossingSignal': 2, char: 1}))
    return scenarios


def get_gender_scenarios():
    """Generate scenarios for gender bias analysis."""
    return [
        ({'Man': 1}, {'Woman': 1}),
        ({'Boy': 1}, {'Girl': 1}),
        ({'OldMan': 1}, {'OldWoman': 1})
    ]


def get_social_role_scenarios():
    """Generate scenarios for social role bias analysis."""
    scenarios = []
    high_status = ['MaleDoctor', 'FemaleDoctor', 'MaleExecutive', 'FemaleExecutive']
    low_status = ['Homeless', 'Criminal']

    for high in high_status:
        for low in low_status:
            scenarios.append(({high: 1}, {low: 1}))
    return scenarios


def get_age_scenarios():
    """Generate scenarios for age bias analysis."""
    scenarios = []
    young = ['Boy', 'Girl']
    old = ['OldMan', 'OldWoman']

    for y in young:
        for o in old:
            scenarios.append(({y: 1}, {o: 1}))
    return scenarios


def get_species_scenarios():
    """Generate scenarios for species bias analysis."""
    scenarios = []
    animals = ['Dog', 'Cat']
    humans = ['Man', 'Woman', 'Boy', 'Girl']

    for human in humans:
        for animal in animals:
            scenarios.append(({human: 1}, {animal: 1}))
    return scenarios


def plot_layer_head_heatmap(importance_by_bias, save_path='layer_head_importance.png'):
    """
    Create heatmap showing importance of each layer-head combination for each bias.
    """
    bias_types = list(importance_by_bias.keys())
    num_biases = len(bias_types)

    # Determine grid dimensions
    num_layers = 2  # Your model has 2 layers
    num_heads = 2   # Your model has 2 heads

    fig, axes = plt.subplots(2, (num_biases + 1) // 2, figsize=(16, 8))
    axes = axes.flatten()

    for idx, (bias_name, importance) in enumerate(importance_by_bias.items()):
        # Create importance matrix
        importance_matrix = np.zeros((num_layers, num_heads))

        for layer_idx in range(num_layers):
            for head_idx in range(num_heads):
                if layer_idx in importance and head_idx in importance[layer_idx]:
                    importance_matrix[layer_idx, head_idx] = \
                        importance[layer_idx][head_idx]['importance_score']

        # Plot heatmap
        sns.heatmap(
            importance_matrix,
            ax=axes[idx],
            cmap='YlOrRd',
            annot=True,
            fmt='.3f',
            cbar=True,
            xticklabels=[f'Head {i}' for i in range(num_heads)],
            yticklabels=[f'Layer {i}' for i in range(num_layers)]
        )
        axes[idx].set_title(f'{bias_name} Bias', fontsize=12, fontweight='bold')

    # Hide unused subplots
    for idx in range(len(bias_types), len(axes)):
        axes[idx].axis('off')

    plt.suptitle('Layer-Head Importance for Different Biases',
                 fontsize=16, fontweight='bold', y=1.02)
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"\nLayer-head heatmap saved to {save_path}")

    return fig


def plot_attention_flow(analyzer, scenario, scenario_name, save_path='attention_flow.png'):
    """
    Visualize attention patterns for a specific scenario across layers.
    """
    attention_by_layer = analyzer.compute_attention_for_scenario(scenario)

    num_layers = len(attention_by_layer)
    num_heads = attention_by_layer[0].shape[1]

    fig, axes = plt.subplots(num_layers, num_heads, figsize=(12, 8))

    # Token labels (CLS + 23 chars for team 0 + 23 chars for team 1)
    token_labels = ['CLS'] + [f'T0:{c[:3]}' for c in CHARACTERS] + [f'T1:{c[:3]}' for c in CHARACTERS]

    for layer_idx in range(num_layers):
        attn_weights = attention_by_layer[layer_idx][0]  # (num_heads, seq_len, seq_len)

        for head_idx in range(num_heads):
            ax = axes[layer_idx, head_idx] if num_layers > 1 else axes[head_idx]

            # Get CLS token attention
            cls_attn = attn_weights[head_idx, 0, :].cpu().numpy()

            # Plot as bar chart
            positions = np.arange(len(cls_attn))
            ax.bar(positions, cls_attn, alpha=0.7)
            ax.set_title(f'Layer {layer_idx}, Head {head_idx}', fontsize=10)
            ax.set_ylabel('Attention Weight')

            # Only show x-labels on bottom row
            if layer_idx == num_layers - 1:
                ax.set_xticks(positions[::5])
                ax.set_xticklabels(token_labels[::5], rotation=90, fontsize=8)
            else:
                ax.set_xticks([])

    plt.suptitle(f'Attention Patterns: {scenario_name}', fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"Attention flow visualization saved to {save_path}")

    return fig


def plot_bias_emergence_by_layer(importance_by_bias, save_path='bias_emergence_by_layer.png'):
    """
    Show how bias emerges progressively through layers.
    """
    bias_types = list(importance_by_bias.keys())
    num_layers = 2

    fig, ax = plt.subplots(figsize=(12, 6))

    x = np.arange(num_layers)
    width = 0.8 / len(bias_types)

    for idx, (bias_name, importance) in enumerate(importance_by_bias.items()):
        layer_scores = []
        for layer_idx in range(num_layers):
            # Sum importance across all heads in this layer
            layer_score = 0
            if layer_idx in importance:
                for head_idx in importance[layer_idx]:
                    layer_score += importance[layer_idx][head_idx]['importance_score']
            layer_scores.append(layer_score)

        offset = (idx - len(bias_types) / 2) * width + width / 2
        ax.bar(x + offset, layer_scores, width, label=bias_name, alpha=0.8)

    ax.set_xlabel('Layer', fontsize=12, fontweight='bold')
    ax.set_ylabel('Total Importance Score', fontsize=12, fontweight='bold')
    ax.set_title('Bias Emergence Across Transformer Layers', fontsize=14, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels([f'Layer {i}' for i in range(num_layers)])
    ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    ax.grid(axis='y', alpha=0.3)

    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"Bias emergence plot saved to {save_path}")

    return fig


def main():
    print("Loading model...")
    model, device = load_model('best_model.pt')

    print("Initializing attention analyzer...")
    analyzer = AttentionAnalyzer(model, device)

    # Define bias types and their scenario generators
    bias_configs = {
        'Legality': get_legality_scenarios,
        'Gender': get_gender_scenarios,
        'Social Role': get_social_role_scenarios,
        'Age': get_age_scenarios,
        'Species': get_species_scenarios,
    }

    importance_by_bias = {}

    # Analyze each bias type
    for bias_name, scenario_fn in bias_configs.items():
        layer_head_contributions = analyzer.analyze_bias_by_layer_head(
            scenario_fn, bias_name
        )

        importance = analyzer.compute_layer_head_importance(layer_head_contributions)
        importance_by_bias[bias_name] = importance

        # Print results
        print(f"\n{'='*60}")
        print(f"{bias_name} Bias - Layer/Head Importance")
        print('='*60)
        for layer_idx in sorted(importance.keys()):
            for head_idx in sorted(importance[layer_idx].keys()):
                metrics = importance[layer_idx][head_idx]
                print(f"Layer {layer_idx}, Head {head_idx}:")
                print(f"  Attention Variance: {metrics['attention_variance']:.4f}")
                print(f"  Avg Correlation:    {metrics['avg_correlation']:.4f}")
                print(f"  Importance Score:   {metrics['importance_score']:.4f}")

    # Create visualizations
    print("\n" + "="*60)
    print("Generating visualizations...")
    print("="*60)

    plot_layer_head_heatmap(importance_by_bias)
    plot_bias_emergence_by_layer(importance_by_bias)

    # Visualize attention for specific scenarios
    example_scenarios = [
        (({'Man': 1}, {'Woman': 1}), 'Gender: Man vs Woman'),
        (({'MaleDoctor': 1}, {'Homeless': 1}), 'Social Role: Doctor vs Homeless'),
        (({'Boy': 1}, {'OldMan': 1}), 'Age: Boy vs OldMan'),
    ]

    for idx, (scenario, name) in enumerate(example_scenarios):
        plot_attention_flow(
            analyzer, scenario, name,
            save_path=f'attention_flow_{idx}.png'
        )

    print("\n" + "="*60)
    print("Analysis complete!")
    print("="*60)
    print("\nGenerated files:")
    print("  - layer_head_importance.png: Heatmap of layer-head importance per bias")
    print("  - bias_emergence_by_layer.png: How bias emerges through layers")
    print("  - attention_flow_*.png: Attention patterns for specific scenarios")


if __name__ == "__main__":
    main()
