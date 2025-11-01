import torch
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA
from scipy.cluster.hierarchy import dendrogram, linkage
from scipy.spatial.distance import cosine, pdist, squareform
from moral_evaluation import load_model, CHARACTERS


# Character categorization for visualization
CHARACTER_CATEGORIES = {
    # Age groups
    'Young': ['Boy', 'Girl', 'Stroller'],
    'Adult': ['Man', 'Woman', 'Pregnant'],
    'Elderly': ['OldMan', 'OldWoman'],

    # Gender (excluding non-gendered)
    'Male': ['Man', 'Boy', 'OldMan', 'MaleDoctor', 'MaleExecutive', 'MaleAthlete', 'LargeMan'],
    'Female': ['Woman', 'Girl', 'OldWoman', 'Pregnant', 'FemaleDoctor', 'FemaleExecutive', 'FemaleAthlete', 'LargeWoman'],

    # Social roles
    'High Status': ['MaleDoctor', 'FemaleDoctor', 'MaleExecutive', 'FemaleExecutive', 'MaleAthlete', 'FemaleAthlete'],
    'Low Status': ['Homeless', 'Criminal'],
    'Neutral': ['Man', 'Woman', 'Boy', 'Girl', 'OldMan', 'OldWoman'],

    # Species
    'Human': [c for c in CHARACTERS if c not in ['Dog', 'Cat', 'Intervention', 'Barrier', 'CrossingSignal']],
    'Animal': ['Dog', 'Cat'],
    'Context': ['Intervention', 'Barrier', 'CrossingSignal'],
}


def extract_character_embeddings(model):
    """Extract character embeddings from the model."""
    with torch.no_grad():
        # Get character embeddings (these are the raw embeddings before composition)
        char_embeddings = model.character_embedding.weight.cpu().numpy()

    return char_embeddings


def extract_cardinality_embeddings(model):
    """Extract cardinality embeddings (count 0-10)."""
    with torch.no_grad():
        card_embeddings = model.cardinality_embedding.weight.cpu().numpy()

    return card_embeddings


def extract_team_embeddings(model):
    """Extract team embeddings (Team 0 vs Team 1)."""
    with torch.no_grad():
        team_embeddings = model.team_embedding.weight.cpu().numpy()

    return team_embeddings


def compute_similarity_matrix(embeddings):
    """Compute cosine similarity matrix for embeddings."""
    # Normalize embeddings
    norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
    normalized = embeddings / (norms + 1e-8)

    # Compute cosine similarity
    similarity = normalized @ normalized.T

    return similarity


def visualize_embeddings_2d(embeddings, labels, title, save_path, method='tsne', category_type=None):
    """
    Visualize embeddings in 2D using t-SNE or PCA.

    Args:
        embeddings: numpy array of shape (n_samples, n_features)
        labels: list of string labels
        title: plot title
        save_path: path to save figure
        method: 'tsne' or 'pca'
        category_type: if provided, color by category (e.g., 'Age', 'Gender', 'Social roles')
    """
    # Filter out context features for cleaner visualization
    sentient_chars = [c for c in CHARACTERS if c not in ['Intervention', 'Barrier', 'CrossingSignal']]
    sentient_indices = [i for i, c in enumerate(labels) if c in sentient_chars]

    if len(sentient_indices) < len(labels):
        embeddings_filtered = embeddings[sentient_indices]
        labels_filtered = [labels[i] for i in sentient_indices]
    else:
        embeddings_filtered = embeddings
        labels_filtered = labels

    # Apply dimensionality reduction
    if method == 'tsne':
        reducer = TSNE(n_components=2, random_state=42, perplexity=min(30, len(embeddings_filtered)-1))
    else:  # pca
        reducer = PCA(n_components=2, random_state=42)

    coords_2d = reducer.fit_transform(embeddings_filtered)

    # Create figure
    fig, ax = plt.subplots(figsize=(14, 10))

    # Color by category if specified
    if category_type:
        # Build category mapping based on type
        if category_type == 'Young':
            categories = {'Young': CHARACTER_CATEGORIES['Young'],
                         'Adult': CHARACTER_CATEGORIES['Adult'],
                         'Elderly': CHARACTER_CATEGORIES['Elderly']}
        elif category_type == 'High Status':
            categories = {'High Status': CHARACTER_CATEGORIES['High Status'],
                         'Low Status': CHARACTER_CATEGORIES['Low Status'],
                         'Neutral': CHARACTER_CATEGORIES['Neutral']}
        else:
            categories = None

        if categories:
            colors_map = plt.cm.tab10(np.linspace(0, 1, len(categories)))

            for idx, (cat_name, cat_chars) in enumerate(categories.items()):
                mask = [label in cat_chars for label in labels_filtered]
                if any(mask):
                    cat_coords = coords_2d[mask]
                    cat_labels = [labels_filtered[i] for i, m in enumerate(mask) if m]

                    ax.scatter(cat_coords[:, 0], cat_coords[:, 1],
                              c=[colors_map[idx]], label=cat_name,
                              s=200, alpha=0.7, edgecolors='black', linewidth=1.5)

                    # Add labels
                    for i, label in enumerate(cat_labels):
                        ax.annotate(label, (cat_coords[i, 0], cat_coords[i, 1]),
                                   fontsize=9, ha='center', va='bottom',
                                   bbox=dict(boxstyle='round,pad=0.3', facecolor='white', alpha=0.7, edgecolor='none'))

            ax.legend(loc='best', fontsize=11, frameon=True, shadow=True)
    else:
        # Default coloring
        scatter = ax.scatter(coords_2d[:, 0], coords_2d[:, 1],
                           c=range(len(coords_2d)), cmap='viridis',
                           s=200, alpha=0.7, edgecolors='black', linewidth=1.5)

        # Add labels for all points
        for i, label in enumerate(labels_filtered):
            ax.annotate(label, (coords_2d[i, 0], coords_2d[i, 1]),
                       fontsize=9, ha='center', va='bottom',
                       bbox=dict(boxstyle='round,pad=0.3', facecolor='white', alpha=0.7, edgecolor='none'))

    ax.set_xlabel(f'{method.upper()} Component 1', fontsize=12, fontweight='bold')
    ax.set_ylabel(f'{method.upper()} Component 2', fontsize=12, fontweight='bold')
    ax.set_title(title, fontsize=14, fontweight='bold')
    ax.grid(alpha=0.3)

    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"Saved: {save_path}")

    return fig


def plot_similarity_heatmap(similarity_matrix, labels, save_path):
    """Plot heatmap of character similarity."""
    # Filter to sentient characters only
    sentient_chars = [c for c in CHARACTERS if c not in ['Intervention', 'Barrier', 'CrossingSignal']]
    sentient_indices = [i for i, c in enumerate(labels) if c in sentient_chars]

    filtered_sim = similarity_matrix[np.ix_(sentient_indices, sentient_indices)]
    filtered_labels = [labels[i] for i in sentient_indices]

    fig, ax = plt.subplots(figsize=(16, 14))

    sns.heatmap(filtered_sim,
                xticklabels=filtered_labels,
                yticklabels=filtered_labels,
                cmap='RdYlGn',
                center=0,
                annot=False,
                fmt='.2f',
                square=True,
                linewidths=0.5,
                cbar_kws={'label': 'Cosine Similarity'},
                ax=ax)

    ax.set_title('Character Embedding Similarity Matrix', fontsize=16, fontweight='bold', pad=20)
    plt.xticks(rotation=45, ha='right')
    plt.yticks(rotation=0)

    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"Saved: {save_path}")

    return fig


def plot_hierarchical_clustering(embeddings, labels, save_path):
    """Create dendrogram showing hierarchical clustering of characters."""
    # Filter to sentient characters
    sentient_chars = [c for c in CHARACTERS if c not in ['Intervention', 'Barrier', 'CrossingSignal']]
    sentient_indices = [i for i, c in enumerate(labels) if c in sentient_chars]

    filtered_embeddings = embeddings[sentient_indices]
    filtered_labels = [labels[i] for i in sentient_indices]

    # Compute linkage
    distances = pdist(filtered_embeddings, metric='cosine')
    linkage_matrix = linkage(distances, method='ward')

    fig, ax = plt.subplots(figsize=(14, 8))

    dendrogram(linkage_matrix,
               labels=filtered_labels,
               ax=ax,
               leaf_font_size=11,
               color_threshold=0.7*max(linkage_matrix[:, 2]))

    ax.set_title('Hierarchical Clustering of Character Embeddings', fontsize=16, fontweight='bold')
    ax.set_xlabel('Character', fontsize=12, fontweight='bold')
    ax.set_ylabel('Distance', fontsize=12, fontweight='bold')
    plt.xticks(rotation=45, ha='right')

    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"Saved: {save_path}")

    return fig


def test_embedding_arithmetic(embeddings, labels):
    """Test if embeddings support meaningful arithmetic operations."""
    results = []

    char_to_idx = {char: idx for idx, char in enumerate(labels)}

    # Test cases: vector arithmetic analogies
    test_cases = [
        ('MaleDoctor', 'Man', 'Woman', 'FemaleDoctor'),  # Doctor - Male + Female = FemaleDoctor?
        ('MaleExecutive', 'Man', 'Woman', 'FemaleExecutive'),
        ('MaleAthlete', 'Man', 'Woman', 'FemaleAthlete'),
        ('Man', 'Boy', 'Girl', 'Woman'),  # Adult male - Boy + Girl = Adult female?
        ('OldMan', 'Man', 'Woman', 'OldWoman'),
    ]

    for a, b, c, expected in test_cases:
        if all(char in char_to_idx for char in [a, b, c, expected]):
            # Compute: a - b + c
            vec_a = embeddings[char_to_idx[a]]
            vec_b = embeddings[char_to_idx[b]]
            vec_c = embeddings[char_to_idx[c]]
            vec_expected = embeddings[char_to_idx[expected]]

            result_vec = vec_a - vec_b + vec_c

            # Find most similar character to result
            similarities = {}
            for char, idx in char_to_idx.items():
                if char not in ['Intervention', 'Barrier', 'CrossingSignal']:
                    sim = 1 - cosine(result_vec, embeddings[idx])
                    similarities[char] = sim

            # Get top 5 most similar
            top_similar = sorted(similarities.items(), key=lambda x: x[1], reverse=True)[:5]

            # Check if expected is in top predictions
            expected_rank = next((i+1 for i, (char, _) in enumerate(top_similar) if char == expected), None)
            expected_similarity = similarities[expected]

            results.append({
                'test': f'{a} - {b} + {c} = {expected}',
                'top_prediction': top_similar[0][0],
                'top_similarity': top_similar[0][1],
                'expected_rank': expected_rank,
                'expected_similarity': expected_similarity,
                'success': top_similar[0][0] == expected,
                'top_5': top_similar
            })

    return results


def plot_embedding_arithmetic_results(results, save_path):
    """Visualize embedding arithmetic test results."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))

    # Plot 1: Success rate and expected rank
    test_names = [r['test'] for r in results]
    ranks = [r['expected_rank'] if r['expected_rank'] else 20 for r in results]
    colors = ['green' if r['success'] else 'orange' if r['expected_rank'] and r['expected_rank'] <= 3 else 'red'
              for r in results]

    ax1.barh(range(len(test_names)), ranks, color=colors, alpha=0.7, edgecolor='black')
    ax1.set_yticks(range(len(test_names)))
    ax1.set_yticklabels([t.replace(' - ', '\n-\n').replace(' + ', '\n+\n').replace(' = ', '\n=\n') for t in test_names],
                         fontsize=9)
    ax1.set_xlabel('Rank of Expected Answer', fontsize=12, fontweight='bold')
    ax1.set_title('Embedding Arithmetic: Expected Answer Rank', fontsize=14, fontweight='bold')
    ax1.axvline(x=1, color='green', linestyle='--', alpha=0.5, label='Rank 1')
    ax1.axvline(x=3, color='orange', linestyle='--', alpha=0.5, label='Top 3')
    ax1.legend()
    ax1.invert_yaxis()
    ax1.grid(axis='x', alpha=0.3)

    # Plot 2: Similarity scores
    expected_sims = [r['expected_similarity'] for r in results]
    top_sims = [r['top_similarity'] for r in results]

    x = np.arange(len(test_names))
    width = 0.35

    ax2.bar(x - width/2, expected_sims, width, label='Expected Answer Similarity', alpha=0.7, color='steelblue')
    ax2.bar(x + width/2, top_sims, width, label='Top Prediction Similarity', alpha=0.7, color='coral')

    ax2.set_ylabel('Cosine Similarity', fontsize=12, fontweight='bold')
    ax2.set_title('Embedding Arithmetic: Similarity Scores', fontsize=14, fontweight='bold')
    ax2.set_xticks(x)
    ax2.set_xticklabels(range(1, len(test_names)+1))
    ax2.set_xlabel('Test Number', fontsize=12, fontweight='bold')
    ax2.legend()
    ax2.grid(axis='y', alpha=0.3)

    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"Saved: {save_path}")

    return fig


def plot_cardinality_embeddings(card_embeddings, save_path):
    """Visualize how cardinality (counts) are encoded."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))

    # Plot 1: Heatmap of cardinality embeddings
    sns.heatmap(card_embeddings.T,
                cmap='viridis',
                xticklabels=range(len(card_embeddings)),
                yticklabels=[f'Dim {i}' for i in range(card_embeddings.shape[1])],
                cbar_kws={'label': 'Embedding Value'},
                ax=ax1)
    ax1.set_title('Cardinality Embeddings (0-10)', fontsize=14, fontweight='bold')
    ax1.set_xlabel('Count', fontsize=12, fontweight='bold')
    ax1.set_ylabel('Embedding Dimension', fontsize=12, fontweight='bold')

    # Plot 2: PCA of cardinality embeddings
    if card_embeddings.shape[0] > 2:
        pca = PCA(n_components=2)
        card_2d = pca.fit_transform(card_embeddings)

        ax2.scatter(card_2d[:, 0], card_2d[:, 1], s=200, c=range(len(card_2d)),
                   cmap='viridis', edgecolors='black', linewidth=1.5)

        for i, (x, y) in enumerate(card_2d):
            ax2.annotate(str(i), (x, y), fontsize=12, ha='center', va='center',
                        bbox=dict(boxstyle='circle,pad=0.3', facecolor='white', alpha=0.8))

        ax2.set_xlabel(f'PC1 ({pca.explained_variance_ratio_[0]:.1%} var)',
                      fontsize=12, fontweight='bold')
        ax2.set_ylabel(f'PC2 ({pca.explained_variance_ratio_[1]:.1%} var)',
                      fontsize=12, fontweight='bold')
        ax2.set_title('Cardinality Embeddings (PCA)', fontsize=14, fontweight='bold')
        ax2.grid(alpha=0.3)

    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"Saved: {save_path}")

    return fig


def plot_team_embeddings(team_embeddings, save_path):
    """Visualize team embeddings."""
    fig, ax = plt.subplots(figsize=(10, 6))

    # Bar plot of team embedding dimensions
    x = np.arange(team_embeddings.shape[1])
    width = 0.35

    ax.bar(x - width/2, team_embeddings[0], width, label='Team 0', alpha=0.7, color='skyblue')
    ax.bar(x + width/2, team_embeddings[1], width, label='Team 1', alpha=0.7, color='salmon')

    ax.set_xlabel('Embedding Dimension', fontsize=12, fontweight='bold')
    ax.set_ylabel('Embedding Value', fontsize=12, fontweight='bold')
    ax.set_title('Team Embeddings Comparison', fontsize=14, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels([f'Dim {i}' for i in range(team_embeddings.shape[1])])
    ax.legend()
    ax.grid(axis='y', alpha=0.3)

    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"Saved: {save_path}")

    return fig


def main():
    print("Loading model...")
    model, device = load_model('best_model.pt')

    print("\n" + "="*70)
    print("EXTRACTING EMBEDDINGS")
    print("="*70)

    # Extract embeddings
    char_embeddings = extract_character_embeddings(model)
    card_embeddings = extract_cardinality_embeddings(model)
    team_embeddings = extract_team_embeddings(model)

    print(f"Character embeddings shape: {char_embeddings.shape}")
    print(f"Cardinality embeddings shape: {card_embeddings.shape}")
    print(f"Team embeddings shape: {team_embeddings.shape}")

    # Compute similarity
    similarity_matrix = compute_similarity_matrix(char_embeddings)

    print("\n" + "="*70)
    print("GENERATING VISUALIZATIONS")
    print("="*70)

    # 1. t-SNE visualizations with different category colorings
    visualize_embeddings_2d(char_embeddings, CHARACTERS,
                           'Character Embeddings (t-SNE) - Colored by Age',
                           'embedding_tsne_age.png', method='tsne',
                           category_type='Young')

    visualize_embeddings_2d(char_embeddings, CHARACTERS,
                           'Character Embeddings (t-SNE) - Colored by Social Role',
                           'embedding_tsne_social.png', method='tsne',
                           category_type='High Status')

    # 2. PCA visualization
    visualize_embeddings_2d(char_embeddings, CHARACTERS,
                           'Character Embeddings (PCA)',
                           'embedding_pca.png', method='pca')

    # 3. Similarity heatmap
    plot_similarity_heatmap(similarity_matrix, CHARACTERS,
                           'embedding_similarity_heatmap.png')

    # 4. Hierarchical clustering
    plot_hierarchical_clustering(char_embeddings, CHARACTERS,
                                'embedding_hierarchical_clustering.png')

    # 5. Embedding arithmetic
    print("\n" + "="*70)
    print("TESTING EMBEDDING ARITHMETIC")
    print("="*70)

    arithmetic_results = test_embedding_arithmetic(char_embeddings, CHARACTERS)

    for result in arithmetic_results:
        print(f"\nTest: {result['test']}")
        print(f"  Top prediction: {result['top_prediction']} (similarity: {result['top_similarity']:.4f})")
        print(f"  Expected rank: {result['expected_rank']}")
        print(f"  Success: {result['success']}")
        print(f"  Top 5: {[(c, f'{s:.3f}') for c, s in result['top_5']]}")

    success_rate = sum(r['success'] for r in arithmetic_results) / len(arithmetic_results)
    top3_rate = sum(1 for r in arithmetic_results if r['expected_rank'] and r['expected_rank'] <= 3) / len(arithmetic_results)

    print(f"\nOverall Success Rate (Rank 1): {success_rate:.1%}")
    print(f"Top-3 Success Rate: {top3_rate:.1%}")

    plot_embedding_arithmetic_results(arithmetic_results,
                                     'embedding_arithmetic_results.png')

    # 6. Cardinality embeddings
    plot_cardinality_embeddings(card_embeddings,
                               'embedding_cardinality.png')

    # 7. Team embeddings
    plot_team_embeddings(team_embeddings,
                        'embedding_team.png')

    print("\n" + "="*70)
    print("ANALYSIS COMPLETE")
    print("="*70)
    print("\nGenerated files:")
    print("  - embedding_tsne_age.png: t-SNE colored by age groups")
    print("  - embedding_tsne_social.png: t-SNE colored by social roles")
    print("  - embedding_pca.png: PCA visualization")
    print("  - embedding_similarity_heatmap.png: Character similarity matrix")
    print("  - embedding_hierarchical_clustering.png: Dendrogram of character groupings")
    print("  - embedding_arithmetic_results.png: Embedding arithmetic test results")
    print("  - embedding_cardinality.png: Cardinality encoding analysis")
    print("  - embedding_team.png: Team embedding comparison")


if __name__ == "__main__":
    main()
