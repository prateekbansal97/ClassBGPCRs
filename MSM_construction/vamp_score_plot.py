import pickle
import matplotlib.pyplot as plt
from tqdm import tqdm
import numpy as np

# Customizing Matplotlib globally
plt.rc('lines', linewidth=1.5)
plt.rc('axes', prop_cycle=plt.cycler('color', ['#FF6F61', '#6B5B95', '#88B04B', '#F7CAC9', '#92A8D1']))
plt.rc('font', size=12)
plt.rc('axes', linewidth=1.5, titlesize=12, spines=dict(top=False, right=False))
plt.rc('xtick.major', width=1.5)
plt.rc('ytick.major', width=1.5)
plt.rc('legend', frameon=False)


class VAMPScoreAnalysis:
    """
    Class to analyze VAMP scores for protein systems.

    Attributes:
        cutoffs (list): List of cutoff values.
        clusters (list): List of cluster sizes.
        proteins (list): List of protein systems.
        minx, maxx (int): X-axis bounds.
        miny, maxy (float): Y-axis bounds for VAMP score plots.
    """
    def __init__(self, cutoffs, clusters, proteins, minx, maxx, miny, maxy):
        self.cutoffs = cutoffs
        self.clusters = clusters
        self.proteins = proteins
        self.minx = minx
        self.maxx = maxx
        self.miny = miny
        self.maxy = maxy

    def load_vamp_score(self, filepath):
        """
        Load VAMP scores from a pickle file.

        Args:
            filepath (str): Path to the pickle file.

        Returns:
            VAMP score data (float).
        """
        return pickle.load(open(filepath, 'rb'))

    def collect_vamp_scores(self, sys, exclude_protein='CALCR'):
        """
        Collect VAMP scores for a given protein system.

        Args:
            sys (str): Protein system under consideration.
            exclude_protein (str): Protein to exclude from the analysis.

        Returns:
            dict: VAMP scores for all proteins.
        """
        if sys == exclude_protein:
            return None

        other_proteins = [prot for prot in self.proteins if prot != sys]
        #print(sys, other_proteins)
        vamp_scores = {prot: {ct: [] for ct in self.cutoffs} for prot in self.proteins}

        for ct in self.cutoffs:
            for k in self.clusters:
                vamp_scores[sys][ct].append(
                    self.load_vamp_score(f'/home/pdb3/ClassB/Analysis/CA_MSM/CA_VAMPscore/VAMPscore_base_{sys}_{k}_{ct}.pkl')
                )
                for other_sys in other_proteins:
                    vamp_scores[other_sys][ct].append(
                        self.load_vamp_score(f'/home/pdb3/ClassB/Analysis/CA_MSM/CA_VAMPscore/VAMPscore_base_{sys}_actual_{other_sys}_{k}_{ct}.pkl')
                )

        return vamp_scores

    def compute_mean_vamp_scores(self, vamp_scores):
        """
        Compute mean VAMP scores across cutoffs and proteins.

        Args:
            vamp_scores (dict): VAMP score data.

        Returns:
            np.array: Mean VAMP scores for plotting.
        """
        mean_score_collected = []
        for prot_scores in vamp_scores.values():
            mean_scores = [np.mean(np.array(scores), axis=1) for scores in prot_scores.values()]
            mean_score_collected.append(np.array(mean_scores))
        return np.mean(np.array(mean_score_collected), axis=0)

    def plot_vamp_scores(self, sys, vamp_scores, mean_score_base):
        """
        Generate and save VAMP score plots.

        Args:
            sys (str): Protein system under consideration.
            vamp_scores (dict): VAMP scores for all proteins.
            mean_score_base (np.array): Mean VAMP scores for the base protein.
        """
        fig, axs = plt.subplots(2, 3, figsize=(612 / 72, 792 / 144))
        best_tic = {}
        max_indices = np.where(mean_score_base == np.max(mean_score_base))

        for ax, prot in zip(axs.flatten(), self.proteins):
            total_mean_scores = []
            for ct in self.cutoffs:
                data = np.array(vamp_scores[prot][ct])
                mean_score = np.mean(data, axis=1)
                stdev_score = np.std(data, axis=1)

                ax.plot(self.clusters, mean_score, label=ct)
                ax.fill_between(self.clusters, mean_score + stdev_score, mean_score - stdev_score, alpha=0.5)
                ax.set_ylim(self.miny, self.maxy)
                ax.set_xlim(self.minx, self.maxx)
                ax.set_title(prot)

                total_mean_scores.append(mean_score)

            clus = self.clusters[max_indices[1][0]]
            tic = self.cutoffs[max_indices[0][0]]
            best_tic[prot] = [clus, tic]

            max_score = np.max(total_mean_scores)
            ax.axvline(clus, linestyle='--', color='k')
            ax.axhline(max_score, linestyle='--', color='k')
            ax.text(clus, max_score, f'({clus}, {tic})', fontsize=8, color='black')

        # Save the best tic-cluster data
        pickle.dump(best_tic, open(f'./CA_optimized_tic_clus/best_tic_clus_base_{sys}.pkl', 'wb'))

        # Save the plot
        handles, labels = ax.get_legend_handles_labels()
        fig.legend(handles, [f'{ct} tICs' for ct in self.cutoffs], loc='upper center', ncol=5, bbox_to_anchor=(0.5, 0.96))
        fig.text(0.5, 0.965, f'Base System: {sys}, Average Score = {np.max(mean_score_base):.2f}', ha='center')
        fig.tight_layout(rect=[0, 0, 1, 0.92])
        fig.savefig(f'./CA_VAMPscore_plots/{sys}_vampscoreplot.png', dpi=300)

    def run_analysis(self):
        """
        Run the full VAMP score analysis for all proteins.
        """
        for sys in tqdm(self.proteins):
            vamp_scores = self.collect_vamp_scores(sys)
            if not vamp_scores:
                continue
            mean_score_base = self.compute_mean_vamp_scores(vamp_scores)
            self.plot_vamp_scores(sys, vamp_scores, mean_score_base)


if __name__ == "__main__":
    cutoffs = [2, 3, 4, 5, 6]
    proteins = ['GCGR', 'GLP1R', 'PAC1R', 'PTH1R', 'SCTR', 'CALCR']
    clusters = [100, 150, 200, 250, 300, 350, 400, 450, 500, 600, 700, 800, 900, 1000, 1200, 1400, 1600, 1800, 2000]
    analysis = VAMPScoreAnalysis(cutoffs, clusters, proteins, minx=100, maxx=2050, miny=3.2, maxy=5.0)
    analysis.run_analysis()
