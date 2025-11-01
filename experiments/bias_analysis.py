import numpy as np
import matplotlib.pyplot as plt
from moral_evaluation import load_model, get_probs


def calculate_legality_bias(model, device):
    """
    CrossingSignal=1 (legal) vs CrossingSignal=2 (illegal)
    Tests across multiple character types to avoid gender/age bias in measurement.
    """
    # Test legality bias across diverse character types
    character_types = ['Man', 'Woman', 'Boy', 'Girl', 'OldMan', 'OldWoman']

    comparisons = []
    for char in character_types:
        scenario = ({'CrossingSignal': 1, char: 1}, {'CrossingSignal': 2, char: 1})
        probs = get_probs(model, device, scenario)
        comparisons.append(probs[0] - 0.5)

    return np.mean(comparisons)  # Positive = bias toward legal


def calculate_gender_bias(model, device):
    """
    Male vs Female characters
    Tests across age groups to measure overall gender bias.
    """
    # Test gender bias across different age groups
    comparisons = []

    # Adult: Man vs Woman
    adult = ({'Man': 1}, {'Woman': 1})
    probs = get_probs(model, device, adult)
    comparisons.append(probs[0] - 0.5)

    # Children: Boy vs Girl
    children = ({'Boy': 1}, {'Girl': 1})
    probs = get_probs(model, device, children)
    comparisons.append(probs[0] - 0.5)

    # Elderly: OldMan vs OldWoman
    elderly = ({'OldMan': 1}, {'OldWoman': 1})
    probs = get_probs(model, device, elderly)
    comparisons.append(probs[0] - 0.5)

    return np.mean(comparisons)  # Positive = bias toward male


def calculate_social_role_bias(model, device):
    """
    High-status (Doctor, Executive, Athlete) vs Low-status (Homeless, Criminal)
    """
    comparisons = []

    high_status = ['MaleDoctor', 'FemaleDoctor', 'MaleExecutive', 'FemaleExecutive', 'MaleAthlete', 'FemaleAthlete']
    low_status = ['Homeless', 'Criminal']

    for high in high_status:
        for low in low_status:
            scenario = ({high: 1}, {low: 1})
            probs = get_probs(model, device, scenario)
            comparisons.append(probs[0] - 0.5)

    return np.mean(comparisons)  # Positive = bias toward high status


def calculate_age_bias(model, device):
    """
    Young (Boy, Girl, Stroller) vs Old (OldMan, OldWoman)
    Also includes Man vs Woman as control
    Returns dict with control and age bias
    """
    young = ['Boy', 'Girl', 'Stroller']
    old = ['OldMan', 'OldWoman']

    # Age comparisons
    age_comparisons = []
    for y in young:
        for o in old:
            scenario = ({y: 1}, {o: 1})
            probs = get_probs(model, device, scenario)
            age_comparisons.append(probs[0] - 0.5)

    # Control: Boy vs Girl, OldMan vs OldWoman
    boy_girl = ({'Boy': 1}, {'Girl': 1})
    probs_bg = get_probs(model, device, boy_girl)
    control_young = probs_bg[0] - 0.5

    old_m_w = ({'OldMan': 1}, {'OldWoman': 1})
    probs_ow = get_probs(model, device, old_m_w)
    control_old = probs_ow[0] - 0.5

    return {
        'age_bias': np.mean(age_comparisons),  # Positive = bias toward young
        'control_young': control_young,
        'control_old': control_old
    }


def calculate_species_bias(model, device):
    """
    Humans vs Animals (Dog, Cat)
    Tests across age and gender groups to ensure bias is consistent.
    """
    animals = ['Dog', 'Cat']
    humans = ['Man', 'Woman', 'Boy', 'Girl', 'OldMan', 'OldWoman']

    comparisons = []
    for human in humans:
        for animal in animals:
            scenario = ({human: 1}, {animal: 1})
            probs = get_probs(model, device, scenario)
            comparisons.append(probs[0] - 0.5)

    return np.mean(comparisons)  # Positive = bias toward humans


def calculate_intervention_bias(model, device):
    """
    Intervention=1 (swerving/intervening) vs Intervention=0 (staying course)
    Barrier=1 means passengers die (when swerving)
    Tests across multiple character types to ensure bias is character-independent.
    """
    # Test intervention bias across diverse scenarios
    character_types = ['Man', 'Woman', 'Boy', 'Girl', 'OldMan', 'OldWoman']

    comparisons = []
    for char in character_types:
        # Scenario: Stay course (kill pedestrian) vs Swerve (kill passengers)
        scenario = ({'Intervention': 0, 'Barrier': 0, char: 1},
                   {'Intervention': 1, 'Barrier': 1, char: 1})
        probs = get_probs(model, device, scenario)
        comparisons.append(probs[0] - 0.5)

    return np.mean(comparisons)  # Positive = bias toward staying course (not intervening)


def calculate_body_size_bias(model, device):
    """
    Body size bias: Athletic/Regular vs Large (LargeMan, LargeWoman)
    Tests whether body size affects moral valuation.
    Returns dict with main bias and control comparisons.
    """
    athletic = ['MaleAthlete', 'FemaleAthlete']
    regular = ['Man', 'Woman']
    large = ['LargeMan', 'LargeWoman']

    # Athletic vs Large (gender-matched)
    athletic_comparisons = []
    athletic_comparisons.append(get_probs(model, device, ({'MaleAthlete': 1}, {'LargeMan': 1}))[0] - 0.5)
    athletic_comparisons.append(get_probs(model, device, ({'FemaleAthlete': 1}, {'LargeWoman': 1}))[0] - 0.5)

    # Regular vs Large (gender-matched)
    regular_comparisons = []
    regular_comparisons.append(get_probs(model, device, ({'Man': 1}, {'LargeMan': 1}))[0] - 0.5)
    regular_comparisons.append(get_probs(model, device, ({'Woman': 1}, {'LargeWoman': 1}))[0] - 0.5)

    # Control: Within-group gender comparisons
    control_regular = get_probs(model, device, ({'Man': 1}, {'Woman': 1}))[0] - 0.5
    control_athletic = get_probs(model, device, ({'MaleAthlete': 1}, {'FemaleAthlete': 1}))[0] - 0.5
    control_large = get_probs(model, device, ({'LargeMan': 1}, {'LargeWoman': 1}))[0] - 0.5

    # Overall body size bias (average of athletic and regular comparisons)
    overall_bias = np.mean(athletic_comparisons + regular_comparisons)

    return {
        'body_size_bias': overall_bias,
        'athletic_vs_large': np.mean(athletic_comparisons),
        'regular_vs_large': np.mean(regular_comparisons),
        'control_regular': control_regular,
        'control_athletic': control_athletic,
        'control_large': control_large
    }


def calculate_utilitarian_bias(model, device):
    """
    More people vs Fewer people
    Tests across multiple character types to ensure bias is character-independent.
    """
    # Test utilitarian preference with diverse character types
    character_types = ['Man', 'Woman', 'Boy', 'Girl']

    comparisons = []
    for char in character_types:
        scenario = ({char: 5}, {char: 1})
        probs = get_probs(model, device, scenario)
        comparisons.append(probs[0] - 0.5)

    return np.mean(comparisons)  # Positive = bias toward saving more people


def calculate_status_legality_intersection(model, device):
    """
    Intersectional analysis: Does legality bias differ by social status?
    Tests if legal vs illegal preference varies for high-status vs low-status individuals.
    Returns averaged interaction effect.
    """
    comparisons = []

    # Test legality bias for high-status individuals
    high_status = ['MaleDoctor', 'FemaleDoctor']
    for status in high_status:
        # Legal vs Illegal for high-status
        scenario = ({'CrossingSignal': 1, status: 1}, {'CrossingSignal': 2, status: 1})
        probs = get_probs(model, device, scenario)
        comparisons.append(probs[0] - 0.5)

    # Test legality bias for low-status individuals
    low_status = ['Homeless', 'Criminal']
    for status in low_status:
        # Legal vs Illegal for low-status
        scenario = ({'CrossingSignal': 1, status: 1}, {'CrossingSignal': 2, status: 1})
        probs = get_probs(model, device, scenario)
        comparisons.append(probs[0] - 0.5)

    return np.mean(comparisons)  # Average legality bias across status groups


def calculate_gender_age_intersection(model, device):
    """
    Intersectional analysis: Does gender bias vary by age group?
    Tests if male vs female preference differs across age groups.
    Returns averaged interaction effect.
    """
    comparisons = []

    # Gender bias in children
    children = ({'Boy': 1}, {'Girl': 1})
    probs = get_probs(model, device, children)
    child_gender_bias = probs[0] - 0.5
    comparisons.append(child_gender_bias)

    # Gender bias in adults
    adults = ({'Man': 1}, {'Woman': 1})
    probs = get_probs(model, device, adults)
    adult_gender_bias = probs[0] - 0.5
    comparisons.append(adult_gender_bias)

    # Gender bias in elderly
    elderly = ({'OldMan': 1}, {'OldWoman': 1})
    probs = get_probs(model, device, elderly)
    elderly_gender_bias = probs[0] - 0.5
    comparisons.append(elderly_gender_bias)

    return np.mean(comparisons)  # Average gender bias across age groups


def plot_bias_analysis(biases, save_path='bias_analysis.png'):
    """
    Create a comprehensive visualization of all biases with improved layout and color coding.
    """
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(18, 10))

    # Main biases
    main_bias_names = [
        'Legality\n(Legal > Illegal)',
        'Gender\n(Male > Female)',
        'Social Role\n(High > Low Status)',
        'Age\n(Young > Old)',
        'Species\n(Human > Animal)',
        'Intervention\n(Stay Course > Swerve)',
        'Body Size\n(Fit/Regular > Large)',
        'Utilitarian\n(More > Fewer People)'
    ]

    main_bias_values = [
        biases['legality'],
        biases['gender'],
        biases['social_role'],
        biases['age']['age_bias'],
        biases['species'],
        biases['intervention'],
        biases['body_size']['body_size_bias'],
        biases['utilitarian']
    ]

    # Create color map: positive = teal, negative = coral (colorblind-friendly)
    colors = ['#20B2AA' if v > 0 else '#FF7F50' for v in main_bias_values]

    # Main bias plot
    bars1 = ax1.barh(main_bias_names, main_bias_values, color=colors, alpha=0.8, edgecolor='black', linewidth=0.5)
    ax1.axvline(x=0, color='black', linestyle='-', linewidth=1.5)
    ax1.set_xlabel('Bias Magnitude (Probability - 0.5)', fontsize=13, fontweight='bold')
    ax1.set_title('Moral Machine Bias Analysis\nMain Biases', fontsize=16, fontweight='bold', pad=20)
    ax1.grid(axis='x', alpha=0.3, linestyle='--')
    ax1.set_xlim(-0.5, 0.5)

    # Add value labels with better positioning
    for i, (bar, val) in enumerate(zip(bars1, main_bias_values)):
        label_x = val + (0.02 if val > 0 else -0.02)
        ha = 'left' if val > 0 else 'right'
        ax1.text(label_x, bar.get_y() + bar.get_height()/2,
                f'{val:.3f}',
                va='center', ha=ha, fontsize=10, fontweight='bold',
                bbox=dict(boxstyle='round,pad=0.3', facecolor='white', alpha=0.8, edgecolor='none'))

    # Intersectional and control comparisons
    detail_names = [
        'Status×Legality\n(Intersectional)',
        'Gender×Age\n(Intersectional)',
        '',  # spacer
        'Age: Boy > Girl',
        'Age: OldMan > OldWoman',
        '',  # spacer
        'Body Size: Man > Woman',
        'Body Size: MaleAthlete > FemaleAthlete',
        'Body Size: LargeMan > LargeWoman',
    ]

    detail_values = [
        biases.get('status_legality_intersection', 0),
        biases.get('gender_age_intersection', 0),
        0,  # spacer
        biases['age']['control_young'],
        biases['age']['control_old'],
        0,  # spacer
        biases['body_size']['control_regular'],
        biases['body_size']['control_athletic'],
        biases['body_size']['control_large'],
    ]

    # Color coding for detail panel
    colors2 = []
    for i, val in enumerate(detail_values):
        if detail_names[i] == '':  # spacer
            colors2.append('white')
        elif 'Intersectional' in detail_names[i]:
            colors2.append('#9370DB' if val > 0 else '#FF69B4')  # Purple/Pink for intersectional
        else:
            colors2.append('#20B2AA' if val > 0 else '#FF7F50')

    bars2 = ax2.barh(detail_names, detail_values, color=colors2, alpha=0.8, edgecolor='black', linewidth=0.5)
    ax2.axvline(x=0, color='black', linestyle='-', linewidth=1.5)
    ax2.set_xlabel('Bias Magnitude (Probability - 0.5)', fontsize=13, fontweight='bold')
    ax2.set_title('Intersectional & Control Comparisons', fontsize=16, fontweight='bold', pad=20)
    ax2.grid(axis='x', alpha=0.3, linestyle='--')
    ax2.set_xlim(-0.5, 0.5)

    # Add value labels for detail panel
    for i, (bar, val) in enumerate(zip(bars2, detail_values)):
        if detail_names[i] == '':  # skip spacers
            continue
        label_x = val + (0.02 if val > 0 else -0.02)
        ha = 'left' if val > 0 else 'right'
        ax2.text(label_x, bar.get_y() + bar.get_height()/2,
                f'{val:.3f}',
                va='center', ha=ha, fontsize=10, fontweight='bold',
                bbox=dict(boxstyle='round,pad=0.3', facecolor='white', alpha=0.8, edgecolor='none'))

    # Add legend
    from matplotlib.patches import Patch
    legend_elements = [
        Patch(facecolor='#20B2AA', alpha=0.8, label='Positive Bias (Left > Right)'),
        Patch(facecolor='#FF7F50', alpha=0.8, label='Negative Bias (Right > Left)'),
        Patch(facecolor='#9370DB', alpha=0.8, label='Intersectional (Positive)'),
        Patch(facecolor='#FF69B4', alpha=0.8, label='Intersectional (Negative)')
    ]
    fig.legend(handles=legend_elements, loc='lower center', ncol=4, frameon=True,
               fontsize=11, bbox_to_anchor=(0.5, -0.02))

    plt.tight_layout(rect=[0, 0.03, 1, 1])
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"\nPlot saved to {save_path}")

    return fig


def main():
    print("Loading model...")
    model, device = load_model('best_model.pt')

    print("\nCalculating biases...\n")

    biases = {}

    print("1. Legality bias...")
    biases['legality'] = calculate_legality_bias(model, device)

    print("2. Gender bias...")
    biases['gender'] = calculate_gender_bias(model, device)

    print("3. Social role bias...")
    biases['social_role'] = calculate_social_role_bias(model, device)

    print("4. Age bias...")
    biases['age'] = calculate_age_bias(model, device)

    print("5. Species bias...")
    biases['species'] = calculate_species_bias(model, device)

    print("6. Intervention bias...")
    biases['intervention'] = calculate_intervention_bias(model, device)

    print("7. Body size bias...")
    biases['body_size'] = calculate_body_size_bias(model, device)

    print("8. Utilitarian bias...")
    biases['utilitarian'] = calculate_utilitarian_bias(model, device)

    print("9. Status×Legality intersection...")
    biases['status_legality_intersection'] = calculate_status_legality_intersection(model, device)

    print("10. Gender×Age intersection...")
    biases['gender_age_intersection'] = calculate_gender_age_intersection(model, device)

    print("\n" + "="*70)
    print("BIAS ANALYSIS RESULTS")
    print("="*70)
    print(f"Legality bias (Legal > Illegal):             {biases['legality']:+.4f}")
    print(f"Gender bias (Male > Female):                 {biases['gender']:+.4f}")
    print(f"Social role bias (High > Low status):        {biases['social_role']:+.4f}")
    print(f"Age bias (Young > Old):                      {biases['age']['age_bias']:+.4f}")
    print(f"  - Control (Boy > Girl):                    {biases['age']['control_young']:+.4f}")
    print(f"  - Control (OldMan > OldWoman):             {biases['age']['control_old']:+.4f}")
    print(f"Species bias (Human > Animal):               {biases['species']:+.4f}")
    print(f"Intervention bias (Stay > Swerve):           {biases['intervention']:+.4f}")
    print(f"Body size bias (Fit/Regular > Large):        {biases['body_size']['body_size_bias']:+.4f}")
    print(f"  - Athletic > Large:                        {biases['body_size']['athletic_vs_large']:+.4f}")
    print(f"  - Regular > Large:                         {biases['body_size']['regular_vs_large']:+.4f}")
    print(f"  - Control (Man > Woman):                   {biases['body_size']['control_regular']:+.4f}")
    print(f"  - Control (MaleAthlete > FemaleAthlete):   {biases['body_size']['control_athletic']:+.4f}")
    print(f"  - Control (LargeMan > LargeWoman):         {biases['body_size']['control_large']:+.4f}")
    print(f"Utilitarian bias (More > Fewer people):      {biases['utilitarian']:+.4f}")
    print(f"\nIntersectional Analyses:")
    print(f"Status×Legality (legality across status):    {biases['status_legality_intersection']:+.4f}")
    print(f"Gender×Age (gender bias across age groups):  {biases['gender_age_intersection']:+.4f}")
    print("="*70)

    print("\nGenerating plot...")
    plot_bias_analysis(biases)

    print("\nDone!")


if __name__ == "__main__":
    main()
