import numpy as np
import matplotlib.pyplot as plt
from moral_evaluation import load_model, get_probs


def calculate_legality_bias(model, device):
    """
    CrossingSignal=1 (legal) vs CrossingSignal=2 (illegal)
    """
    # Legal pedestrians vs illegal pedestrians
    legal = ({'CrossingSignal': 1, 'Man': 1}, {'CrossingSignal': 2, 'Man': 1})
    probs = get_probs(model, device, legal)
    return probs[0] - 0.5  # Positive = bias toward legal


def calculate_gender_bias(model, device):
    """
    Male vs Female characters
    """
    # Man vs Woman
    male_vs_female = ({'Man': 1}, {'Woman': 1})
    probs = get_probs(model, device, male_vs_female)
    return probs[0] - 0.5  # Positive = bias toward male


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
    """
    animals = ['Dog', 'Cat']
    humans = ['Man', 'Woman']

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
    """
    # Scenario: Stay course (kill 1 pedestrian) vs Swerve (kill passengers)
    # Intervention=0, Barrier=0: pedestrian dies
    # Intervention=1, Barrier=1: passenger dies
    stay_course = ({'Intervention': 0, 'Barrier': 0, 'Man': 1},
                   {'Intervention': 1, 'Barrier': 1, 'Man': 1})

    probs = get_probs(model, device, stay_course)
    return probs[0] - 0.5  # Positive = bias toward staying course (not intervening)


def calculate_fitness_bias(model, device):
    """
    Athletic/Regular vs Large (LargeMan, LargeWoman)
    Also includes Man vs Woman as control
    Returns dict with control and fitness bias
    """
    athletic = ['MaleAthlete', 'FemaleAthlete']
    regular = ['Man', 'Woman']
    large = ['LargeMan', 'LargeWoman']

    # Athletic vs Large
    athletic_comparisons = []
    for ath in athletic:
        for lg in large:
            scenario = ({ath: 1}, {lg: 1})
            probs = get_probs(model, device, scenario)
            athletic_comparisons.append(probs[0] - 0.5)

    # Regular vs Large
    regular_comparisons = []
    for reg in regular:
        for lg in large:
            scenario = ({reg: 1}, {lg: 1})
            probs = get_probs(model, device, scenario)
            regular_comparisons.append(probs[0] - 0.5)

    # Control: Man vs Woman, MaleAthlete vs FemaleAthlete, LargeMan vs LargeWoman
    control_regular = ({'Man': 1}, {'Woman': 1})
    probs_reg = get_probs(model, device, control_regular)

    control_athletic = ({'MaleAthlete': 1}, {'FemaleAthlete': 1})
    probs_ath = get_probs(model, device, control_athletic)

    control_large = ({'LargeMan': 1}, {'LargeWoman': 1})
    probs_lg = get_probs(model, device, control_large)

    return {
        'athletic_vs_large': np.mean(athletic_comparisons),
        'regular_vs_large': np.mean(regular_comparisons),
        'control_regular': probs_reg[0] - 0.5,
        'control_athletic': probs_ath[0] - 0.5,
        'control_large': probs_lg[0] - 0.5
    }


def calculate_utilitarian_bias(model, device):
    """
    More people vs Fewer people
    """
    # 5 people vs 1 person
    many_vs_few = ({'Man': 5}, {'Man': 1})
    probs = get_probs(model, device, many_vs_few)

    return probs[0] - 0.5  # Positive = bias toward saving more people


def plot_bias_analysis(biases, save_path='bias_analysis.png'):
    """
    Create a comprehensive visualization of all biases
    """
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 8))

    # Main biases
    main_bias_names = [
        'Legality\n(Legal > Illegal)',
        'Gender\n(Male > Female)',
        'Social Role\n(High > Low)',
        'Age\n(Young > Old)',
        'Species\n(Human > Animal)',
        'Intervention\n(Stay > Swerve)',
        'Fitness\n(Fit > Large)',
        'Utilitarian\n(More > Fewer)'
    ]

    main_bias_values = [
        biases['legality'],
        biases['gender'],
        biases['social_role'],
        biases['age']['age_bias'],
        biases['species'],
        biases['intervention'],
        biases['fitness']['regular_vs_large'],
        biases['utilitarian']
    ]

    # Create color map: positive = blue, negative = red
    colors = ['#2E86AB' if v > 0 else '#A23B72' for v in main_bias_values]

    # Main bias plot
    bars1 = ax1.barh(main_bias_names, main_bias_values, color=colors, alpha=0.7)
    ax1.axvline(x=0, color='black', linestyle='-', linewidth=1)
    ax1.set_xlabel('Bias Magnitude (Probability - 0.5)', fontsize=12)
    ax1.set_title('Moral Machine Bias Analysis\nMain Biases', fontsize=14, fontweight='bold')
    ax1.grid(axis='x', alpha=0.3)

    # Add value labels
    for i, (bar, val) in enumerate(zip(bars1, main_bias_values)):
        label_x = val + (0.01 if val > 0 else -0.01)
        ha = 'left' if val > 0 else 'right'
        ax1.text(label_x, bar.get_y() + bar.get_height()/2,
                f'{val:.3f}',
                va='center', ha=ha, fontsize=9, fontweight='bold')

    # Control comparisons (Age and Fitness)
    control_names = [
        'Age: Boy > Girl',
        'Age: OldMan > OldWoman',
        'Fitness: Man > Woman',
        'Fitness: M.Ath > F.Ath',
        'Fitness: L.Man > L.Woman',
        'Fitness: Athletic > Large',
    ]

    control_values = [
        biases['age']['control_young'],
        biases['age']['control_old'],
        biases['fitness']['control_regular'],
        biases['fitness']['control_athletic'],
        biases['fitness']['control_large'],
        biases['fitness']['athletic_vs_large']
    ]

    colors2 = ['#2E86AB' if v > 0 else '#A23B72' for v in control_values]

    bars2 = ax2.barh(control_names, control_values, color=colors2, alpha=0.7)
    ax2.axvline(x=0, color='black', linestyle='-', linewidth=1)
    ax2.set_xlabel('Bias Magnitude (Probability - 0.5)', fontsize=12)
    ax2.set_title('Control Comparisons\n(Age & Fitness Detail)', fontsize=14, fontweight='bold')
    ax2.grid(axis='x', alpha=0.3)

    # Add value labels
    for i, (bar, val) in enumerate(zip(bars2, control_values)):
        label_x = val + (0.01 if val > 0 else -0.01)
        ha = 'left' if val > 0 else 'right'
        ax2.text(label_x, bar.get_y() + bar.get_height()/2,
                f'{val:.3f}',
                va='center', ha=ha, fontsize=9, fontweight='bold')

    plt.tight_layout()
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

    print("7. Fitness bias...")
    biases['fitness'] = calculate_fitness_bias(model, device)

    print("8. Utilitarian bias...")
    biases['utilitarian'] = calculate_utilitarian_bias(model, device)

    print("\n" + "="*60)
    print("BIAS ANALYSIS RESULTS")
    print("="*60)
    print(f"Legality bias (Legal > Illegal):           {biases['legality']:+.4f}")
    print(f"Gender bias (Male > Female):               {biases['gender']:+.4f}")
    print(f"Social role bias (High > Low status):      {biases['social_role']:+.4f}")
    print(f"Age bias (Young > Old):                    {biases['age']['age_bias']:+.4f}")
    print(f"  - Control (Boy > Girl):                  {biases['age']['control_young']:+.4f}")
    print(f"  - Control (OldMan > OldWoman):           {biases['age']['control_old']:+.4f}")
    print(f"Species bias (Human > Animal):             {biases['species']:+.4f}")
    print(f"Intervention bias (Stay > Swerve):         {biases['intervention']:+.4f}")
    print(f"Fitness bias (Regular > Large):            {biases['fitness']['regular_vs_large']:+.4f}")
    print(f"  - Athletic > Large:                      {biases['fitness']['athletic_vs_large']:+.4f}")
    print(f"  - Control (Man > Woman):                 {biases['fitness']['control_regular']:+.4f}")
    print(f"  - Control (MaleAthlete > FemaleAthlete): {biases['fitness']['control_athletic']:+.4f}")
    print(f"  - Control (LargeMan > LargeWoman):       {biases['fitness']['control_large']:+.4f}")
    print(f"Utilitarian bias (More > Fewer people):    {biases['utilitarian']:+.4f}")
    print("="*60)

    print("\nGenerating plot...")
    plot_bias_analysis(biases)

    print("\nDone!")


if __name__ == "__main__":
    main()
