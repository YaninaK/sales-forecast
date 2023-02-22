import pandas as pd
import scipy.stats as st
from tqdm import tqdm
import matplotlib.pyplot as plt
import seaborn as sns

DEFAULT_RANDOM_SEED = 25


def get_missing_days_info(missing_days: list):
    years = sorted(list(set([i.year for i in missing_days])))
    print("Date omissions:")
    for year in years:
        n1 = 0
        year_missing_days = [i for i in missing_days if i.year == year]
        if year_missing_days:
            if n1 == 0:
                print(f"\n{year}")
                n1 += 1
            for month in range(1, 13):
                month_missing_days = [i for i in year_missing_days if i.month == month]
                if month_missing_days:
                    n2 = 0
                    for weekday in range(7):
                        weekday_missing_days = [
                            i for i in month_missing_days if i.weekday() == weekday
                        ]
                        if weekday_missing_days:
                            if n2 == 0:
                                print(f'\n{weekday_missing_days[0].strftime("%B")}')
                            weekday_missing_days = [
                                i.date() for i in weekday_missing_days
                            ]
                            print(
                                f' {weekday_missing_days[0].strftime("%a")}:',
                                *weekday_missing_days,
                            )
                            n2 += 1


def get_distribution(data: pd.DataFrame, shop: int):
    target = data[shop]
    fig, axes = plt.subplots(1, 3, figsize=(12, 3))

    axes[0].set_title("Johnson SU", fontsize=14)
    sns.distplot(target, kde=False, color="blue", fit=st.johnsonsu, ax=axes[0])

    axes[1].set_title("Normal", fontsize=14)
    sns.distplot(target, kde=False, color="blue", fit=st.norm, ax=axes[1])

    axes[2].set_title("Log Normal", fontsize=14)
    sns.distplot(target, kde=False, color="blue", fit=st.lognorm, ax=axes[2])
    plt.show()


def get_n_clusters(
    ts_scaled: pd.DataFrame, metric="euclidean", random_state: Optional[int] = None
):
    """
    silhouette_score calculates how clean the clusters are:
    https://scikit-learn.org/stable/modules/generated/sklearn.metrics.silhouette_score.html

    For metric="dtw" tslearn implementation is used:
    https://tslearn.readthedocs.io/en/stable/gen_modules/clustering/tslearn.clustering.silhouette_score.html

    distortions - sum of sq. distances from objects to the center of the cluster, weighted if weights are available
    """
    if random_state is None:
        random_state = DEFAULT_RANDOM_SEED

    distortions = []
    silhouette = []
    K = range(2, 10)
    for k in tqdm(K):
        model = TimeSeriesKMeans(
            n_clusters=k,
            metric=metric,
            n_jobs=6,
            max_iter=10,
            n_init=5,
            random_state=random_state,
        )
        model.fit(ts_scaled)
        distortions.append(model.inertia_)
        silhouette.append(silhouette_score(ts_scaled, model.labels_, metric=metric))

    fig, ax1 = plt.subplots()

    ax2 = ax1.twinx()
    ax1.plot(K, distortions, "b-")
    ax2.plot(K, silhouette, "r-")

    ax1.set_xlabel("# clusters")
    ax1.set_ylabel("Distortion", color="b")
    ax2.set_ylabel("Silhouette", color="r")

    plt.show()


def plot_cluster_centroid(ts_model, n_clusters: int):
    plt.figure(figsize=(12, 5))
    for cluster_number in range(n_clusters):
        cluster_centroid = ts_model.cluster_centers_[cluster_number, :, 0].T
        plt.plot(cluster_centroid, label=cluster_number)
    plt.title("Cluster centroids")
    plt.show()
