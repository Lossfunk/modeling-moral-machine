import numpy as np
import matplotlib.pyplot as plt
from moral_evaluation import load_model, get_probs, CHARACTERS, CHAR_TO_IDX


def get_moral_weight(model, device, char1, char2):
    """
    Compare two characters by calculating their relative moral weight.
    Returns the probability that char1 is chosen over char2.
    """
    scenario = ({char1: 1}, {char2: 1})
    probs = get_probs(model, device, scenario)
    return probs[0]  # Probability of choosing char1 (outcome_0)


def pairwise_comparison(model, device, characters):
    """
    Compare each character with every other character.
    Man is the relative weight of 1.
    """
    # Filter out non-character entities
    actual_characters = [c for c in characters if c not in ['Intervention', 'Barrier', 'CrossingSignal']]

    weights = {}

    # For each character, calculate average preference when compared to all others
    for char in actual_characters:
        total_weight = 0
        comparisons = 0

        for other_char in actual_characters:
            if char != other_char:
                prob = get_moral_weight(model, device, char, other_char)
                total_weight += prob
                comparisons += 1

        # Average weight across all comparisons
        weights[char] = total_weight / comparisons if comparisons > 0 else 0.5

    # Normalize to Man = 1
    man_weight = weights.get('Man', 1.0)
    for char in weights:
        weights[char] = weights[char] / man_weight

    return weights


def plot_pairwise_weights(weights, save_path='moral_weights.png'):
    """
    Create a bar graph of pairwise comparison moral weights.
    """
    # Sort characters by weight
    sorted_chars = sorted(weights.items(), key=lambda x: x[1], reverse=True)
    chars = [c[0] for c in sorted_chars]
    vals = [c[1] for c in sorted_chars]

    # Create figure
    fig, ax = plt.subplots(figsize=(12, 8))

    # Color Man in red, others in skyblue
    colors = ['red' if c == 'Man' else 'skyblue' for c in chars]
    bars = ax.barh(chars, vals, color=colors)

    # Add reference line at Man = 1.0
    ax.axvline(x=1.0, color='red', linestyle='--', linewidth=1, alpha=0.5)

    # Add value labels on bars
    for i, (bar, val) in enumerate(zip(bars, vals)):
        ax.text(val, bar.get_y() + bar.get_height()/2,
                f' {val:.3f}',
                va='center', fontsize=9)

    ax.set_xlabel('Relative Moral Weight', fontsize=12)
    ax.set_title('Pairwise Comparison: Moral Weights\n(Man = 1.0)', fontsize=14, fontweight='bold')
    ax.grid(axis='x', alpha=0.3)

    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"\nPlot saved to {save_path}")

    return fig


def main():
    print("Loading model...")
    model, device = load_model('best_model.pt')

    print("\nCalculating pairwise comparisons...\n")
    weights = pairwise_comparison(model, device, CHARACTERS)

    print("="*60)
    print("MORAL WEIGHTS (Man = 1.0)")
    print("="*60)
    sorted_weights = sorted(weights.items(), key=lambda x: x[1], reverse=True)
    for char, weight in sorted_weights:
        print(f"{char:20s}: {weight:.4f}")

    print("\nGenerating plot...")
    plot_pairwise_weights(weights)

    print("\nDone!")


if __name__ == "__main__":
    main()
