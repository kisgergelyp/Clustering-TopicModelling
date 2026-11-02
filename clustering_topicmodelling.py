import pandas as pd
import os
import numpy as np
import matplotlib.pyplot as plt
from collections import Counter

# Sklearn importok
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.cluster import KMeans, AgglomerativeClustering
from sklearn.decomposition import LatentDirichletAllocation, NMF, TruncatedSVD
from sklearn.metrics import silhouette_score
from sklearn.preprocessing import Normalizer

# Beágyazások és modern klaszterezés
from sentence_transformers import SentenceTransformer
import umap
import hdbscan
from bertopic import BERTopic

# Gensim a koherencia számításhoz
from gensim.corpora import Dictionary
from gensim.models.coherencemodel import CoherenceModel


def main():
    # 1. ADATOK BETÖLTÉSE
    filename = askforvideoid()
    if filename is None:
        # dupla ellenőrzés?
        print("A program leáll, mert nincs meg a fájl.")
        return
    else:
        # A főprogram megpróbálja beolvasni a DataFrame-et
        try:
            df = pd.read_csv(f"tisztitott_adatok/{filename}_tisztazva.csv")
            print("Fájl sikeresen beolvasva!")
        except Exception as e:
            print(f"Váratlan hiba történt a beolvasás során: {e}")
            return

    #teljesen feldolgozott kommentek betöltése
    kommentek = df['clean_final'].tolist()

    # mivel a BERt-nek szüksége van a stopword és a lemmatizálás előtti állapotra, ezért ez is kell:
    kommentek_bert = df['clean2'].astype(str).tolist()

    print(f"Összes tisztított komment: {len(kommentek)}")

    # TF-IDF VEKTORIZÁLÁS (K-Means-hez)
    vectorizer = TfidfVectorizer(max_features=5000, min_df=2, max_df=0.85)
    X = vectorizer.fit_transform(kommentek)
    print(f"TF-IDF mátrix mérete: {X.shape}")

    # 2. KLASZTEREK SZÁMÁNAK MEGHATÁROZÁSA (KÖNYÖK MÓDSZER + SILHOUETTE)
    print("\nKönyök módszer és Silhouette-pontszám számítása (2-15 klaszter)...")
    inertias = []
    silhouettes = []
    K_range = range(2, 16)

    for k in K_range:
        km = KMeans(n_clusters=k, random_state=42, n_init=10)
        labels = km.fit_predict(X)

        inertias.append(km.inertia_)
        # sample_size=5000 felgyorsítja a számítást nagy adatbázisnál,
        # kisebb adatbázisnál automatikusan az összes sort használja
        sil_score = silhouette_score(X, labels, sample_size=min(5000, X.shape[0]), random_state=42)
        silhouettes.append(sil_score)
        print(f"  k={k:2d} | Inercia: {km.inertia_:.1f} | Silhouette: {sil_score:.4f}")

    # Két grafikon egymás mellett (1 sor, 2 oszlop)
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14.0, 5.0))

    # Bal oldali ábra: Könyök-módszer
    ax1.plot(K_range, inertias, marker='o', color='tab:blue')
    ax1.set_xlabel('Klaszterek száma (k)')
    ax1.set_ylabel('Inercia (SSE)')
    ax1.set_title('Könyök módszer (Keresd a töréspontot!)')
    ax1.grid(True, linestyle='--', alpha=0.5)

    # Jobb oldali ábra: Silhouette-pontszám
    ax2.plot(K_range, silhouettes, marker='s', color='tab:orange')
    ax2.set_xlabel('Klaszterek száma (k)')
    ax2.set_ylabel('Silhouette-pontszám')
    ax2.set_title('Silhouette módszer (Keresd a legmagasabb csúcsot!)')
    ax2.grid(True, linestyle='--', alpha=0.5)

    plt.tight_layout()
    plt.show(block=False)
    plt.pause(1.0)

    # Megkeressük a matematikailag legjobb k értéket a Silhouette alapján
    best_k = list(K_range)[np.argmax(silhouettes)]
    print(f"\n A Silhouette-pontszám alapján a {best_k}. klaszter tűnik a legjobbnak.")

    while True:
        try:
            val = input("\nNézd meg a grafikont és add meg a választott klaszterek számát: ")
            N_CLUSTERS = int(val)
            if N_CLUSTERS >= 2:
                break
            else:
                print("A klaszterek számának legalább 2-nek kell lennie!")
        except ValueError:
            print("Érvénytelen bemenet. Kérlek, egy egész számot adj meg!")

    plt.close()

    # K-MEANS FUTTATÁSA
    print(f"\nK-means futtatása {N_CLUSTERS} klaszterrel...")
    kmeans = KMeans(n_clusters=N_CLUSTERS, random_state=42, n_init=20)
    df['cluster'] = kmeans.fit_predict(X)

    # KLASZTEREK ÉRTELMEZÉSE
    print("\nK-MEANS KLASZTEREK TOP SZAVAI ÉS LEGTIPIKUSABB PÉLDÁI:")
    feature_names_kmeans = vectorizer.get_feature_names_out()

    # Kiszámoljuk minden komment távolságát minden klaszterközépponttól
    all_distances = kmeans.transform(X)

    for cluster_id in df['cluster'].value_counts().index:
        # Megkeressük az adott klaszterbe tartozó sorok pozícióit (0, 1, 2...)
        cluster_indices = np.where(df['cluster'].to_numpy() == cluster_id)[0]
        count = len(cluster_indices)

        center = kmeans.cluster_centers_[cluster_id]
        top_indices = np.argsort(center)[-10:][::-1]
        top_words = [feature_names_kmeans[i] for i in top_indices]

        print(f"\n🔹 Klaszter {cluster_id:2d} (db: {count:4d}) | Top szavak: {', '.join(top_words)}")

        # Kiválasztjuk a klaszter középpontjához 3 legközelebbi kommentet
        cluster_distances = all_distances[cluster_indices, cluster_id]
        closest_row_positions = cluster_indices[np.argsort(cluster_distances)[:min(3, count)]]

        # a legtipikusabb szövegeket (hogy ne véletlenszerű páldák legyenek)
        # a nem stopword szűrt és nem lemmtizált sözvegből
        best_examples = df['clean2'].iloc[closest_row_positions].tolist()

        for idx_num, text in enumerate(best_examples, 1):
            pelda_szoveg = str(text).replace('\n', ' ').strip()
            if len(pelda_szoveg) > 200:
                pelda_szoveg = pelda_szoveg[:197] + "..."
            print(f"      {idx_num}. példa: \"{pelda_szoveg}\"")


    # 3. HIERARCHIKUS KLASZTEREZÉS
    print("\n" + "=" * 50)
    print("3. HIERARCHIKUS KLASZTEREZÉS (3000-es minta)...")
    print("=" * 50)

    sample_size = 3000
    sample_df = df.sample(n=min(sample_size, len(df)), random_state=42).copy().reset_index(drop=True)
    # az egyiket a vectozireznek hozom létre, a másikat a példákhoz
    sample_texts = sample_df['clean_final'].tolist()
    sample_texts_readable = sample_df['clean2'].tolist()
    print(f"  Mintanagyság: {len(sample_texts)} komment")

    vectorizer_hier = TfidfVectorizer(max_features=2000, min_df=2, max_df=0.8)
    x_tfidf = vectorizer_hier.fit_transform(sample_texts)

    n_comp = min(100, x_tfidf.shape[1] - 1)
    svd = TruncatedSVD(n_components=n_comp, random_state=42)
    normalizer = Normalizer(copy=False)
    x_hier = normalizer.fit_transform(svd.fit_transform(x_tfidf))

    best_k = 10
    best_score = -np.inf
    test_clusters = [10, 15, 20, 25, 30]

    for k in test_clusters:
        agg_test = AgglomerativeClustering(n_clusters=k, linkage='ward')
        labels = agg_test.fit_predict(x_hier)
        sizes = list(Counter(labels).values())

        if len([s for s in sizes if s < 5]) > k * 0.3:
            continue

        score = silhouette_score(x_hier, labels, metric='euclidean')
        print(f"  Teszt k={k:2d} | Silhouette score: {score:.4f}")

        if score > best_score:
            best_score = score
            best_k = k

    print(f"\n  Optimális klaszterszám a teszt alapján: {best_k}")

    agg = AgglomerativeClustering(n_clusters=best_k, linkage='ward')
    labels_final = agg.fit_predict(x_hier)
    sample_df['cluster_hier'] = labels_final

    if best_score == -np.inf:
        best_score = silhouette_score(x_hier, labels_final, metric='euclidean')

    print(f"\nHierarchikus klaszterek (top szavak és példák, összesen {best_k} klaszter):")
    cluster_sizes = Counter(labels_final).most_common()
    feature_names_hier = np.array(vectorizer_hier.get_feature_names_out())

    for i, (cluster_id, count) in enumerate(cluster_sizes):
        if i >= 15:
            print(f"\n  ... és további {len(cluster_sizes) - 15} klaszter.")
            break

        cluster_indices = np.where(labels_final == cluster_id)[0]

        mean_tfidf = x_tfidf[cluster_indices].mean(axis=0).A1
        top_words = feature_names_hier[mean_tfidf.argsort()[-8:][::-1]]

        print(f"\n🔹 Klaszter {cluster_id:2d} (db: {count:4d}) | Top szavak: {', '.join(top_words)}")

        cluster_vectors = x_hier[cluster_indices]
        centroid = cluster_vectors.mean(axis=0)
        distances = np.linalg.norm(cluster_vectors - centroid, axis=1)

        closest_indices = cluster_indices[distances.argsort()[:min(3, count)]]

        for idx_num, idx in enumerate(closest_indices, 1):
            pelda_szoveg = str(sample_texts_readable[idx]).replace('\n', ' ').strip()
            if len(pelda_szoveg) > 200:
                pelda_szoveg = pelda_szoveg[:197] + "..."
            print(f"     {idx_num}. példa: \"{pelda_szoveg}\"")

    from scipy.cluster.hierarchy import dendrogram, linkage

    print("  > Dendrogram generálása...")
    # noinspection PyTypeChecker
    plt.figure(figsize=(12, 5)) # Téves warning?
    # Ward-linkage mátrix kiszámítása az x_hier mátrixon
    Z = linkage(x_hier, method='ward')

    # Csak a felső 30 ágat (a 30 klasztert) rajzoljuk ki, hogy olvasható legyen
    dendrogram(
        Z,
        truncate_mode='lastp',
        p=best_k,
        leaf_rotation=90.,
        leaf_font_size=9,
        show_contracted=True
    )
    plt.title(f'Hierarchikus klaszterezés dendrogramja (Top {best_k} klaszter, Ward-módszer)')
    plt.xlabel('Klaszter mérete (zárójelben a kommentek száma) vagy indexe')
    plt.ylabel('Euklideszi távolság (Ward)')
    plt.tight_layout()

    results_folder = os.path.join("diagramok", filename)
    os.makedirs(results_folder, exist_ok=True)

    plt.savefig(os.path.join(results_folder, f"{filename}_3_hierarchikus_dendrogram.png"), dpi=300)
    plt.show()


    # 4. EMBEDDING-ALAPÚ KLASZTEREZÉS (BERT + UMAP + HDBSCAN)
    print("\n" + "=" * 50)
    print("4. EMBEDDING-ALAPÚ KLASZTEREZÉS FUTÁSA...")
    print("=" * 50)

    model = SentenceTransformer('all-MiniLM-L6-v2')
    print("  > Embedding-ek generálása...")
    # A nem stopword szűrt és nem lemattizált szöveget adom neki
    embeddings = model.encode(kommentek_bert, show_progress_bar=True, batch_size=64)

    print("  > Dimenziócsökkentés UMAP-pal...")
    umap_reducer = umap.UMAP(n_components=5, random_state=42, n_neighbors=15, min_dist=0.0)
    umap_embeddings = umap_reducer.fit_transform(embeddings)

    print("  > HDBSCAN klaszterezés...")
    clusterer = hdbscan.HDBSCAN(min_cluster_size=15, min_samples=5, prediction_data=True)
    df['cluster_embed'] = clusterer.fit_predict(umap_embeddings)

    # megszámoljuk mennyi klaszter van kivéve a -1 et
    n_clusters_embed = df.loc[df['cluster_embed'] != -1, 'cluster_embed'].nunique()

    #mennyi kommentet nem soroltunk be? (zajos kommentek)
    n_noise = sum(df['cluster_embed'] == -1)

    print(f"\n  Talált klaszterek száma: {n_clusters_embed}")
    print(f"  Zajos (nem besorolt) kommentek: {n_noise}")

    print("\nEmbedding-alapú klaszterek (top szavak és példák):")

    # Méret szerint csökkenő sorrendben íratjuk ki a klasztereket (a -1 zajt kihagyva)
    embed_counts = df[df['cluster_embed'] != -1]['cluster_embed'].value_counts()

    for i, (cluster_id, count) in enumerate(embed_counts.items()):
        if i >= 15:
            print(f"\n  ... és további {len(embed_counts) - 15} klaszter.")
            break

        cluster_indices = np.where(df['cluster_embed'].to_numpy() == cluster_id)[0]

        mean_tfidf = X[cluster_indices].mean(axis=0).A1
        top_words = feature_names_kmeans[mean_tfidf.argsort()[-10:][::-1]]

        print(f"\n🔹 Klaszter {cluster_id:2d} (db: {count:4d}) | Top szavak: {', '.join(top_words)}")

        # A 3 legjellemzőbb komment megkeresése (középponthoz legközelebbiek)
        cluster_vectors = umap_embeddings[cluster_indices]
        centroid = cluster_vectors.mean(axis=0)
        distances = np.linalg.norm(cluster_vectors - centroid, axis=1)

        closest_indices = cluster_indices[distances.argsort()[:min(3, count)]]

        for idx_num, idx in enumerate(closest_indices, 1):
            pelda_szoveg = str(kommentek_bert[idx]).replace('\n', ' ').strip()
            if len(pelda_szoveg) > 200:
                pelda_szoveg = pelda_szoveg[:197] + "..."
            print(f"     {idx_num}. példa: \"{pelda_szoveg}\"")

    # Vizualizáció
    print("  > 2D UMAP vizualizáció készítése...")
    umap_2d = umap.UMAP(n_components=2, random_state=42, n_neighbors=15, min_dist=0.1).fit_transform(embeddings)

    # noinspection PyTypeChecker
    plt.figure(figsize=(12, 8)) #szintén téves warning
    # 1. Zajos (-1) és kisebb klaszterek kirajzolása halványszürkével a háttérben
    top_10_clusters = embed_counts.head(10).index.tolist()
    bg_mask = ~df['cluster_embed'].isin(top_10_clusters)
    plt.scatter(umap_2d[bg_mask, 0], umap_2d[bg_mask, 1], c='lightgray', s=3, alpha=0.3,
                label='Zaj (-1) / Kisebb klaszterek')

    # 2. A 10 legnagyobb klaszter kirajzolása élénk színekkel
    cmap = plt.get_cmap('tab10')
    for idx, cid in enumerate(top_10_clusters):
        c_mask = (df['cluster_embed'] == cid).values
        # Top 3 legjellemzőbb szó lekérése TF-IDF alapján a jelmagyarázathoz
        mean_tfidf_c = X[c_mask].mean(axis=0).A1
        top3_words = feature_names_kmeans[mean_tfidf_c.argsort()[-3:][::-1]]
        top3 = ', '.join(top3_words)

        plt.scatter(umap_2d[c_mask, 0], umap_2d[c_mask, 1], color=cmap(idx), s=8, alpha=0.8,
                    label=f'Klaszter {cid} ({sum(c_mask)} db): {top3}')

    plt.title('Kommentek szemantikai térképe (BERT + 2D UMAP + HDBSCAN Top 10 klaszter)')
    plt.xlabel('UMAP 1. dimenzió')
    plt.ylabel('UMAP 2. dimenzió')
    plt.legend(markerscale=2, fontsize=9, loc='best', framealpha=0.9)
    plt.tight_layout()
    plt.savefig(os.path.join(results_folder, f"{filename}_4_umap_hdbscan_klaszterek.png"), dpi=300)
    plt.show()


    # 5. TELJESÍTMÉNYÉRTÉKELÉS
    print("\n" + "=" * 50)
    print("5. TELJESÍTMÉNYÉRTÉKELÉS (Silhouette, Diverzitás)...")
    print("=" * 50)

    sil_kmeans = 0.0
    try:
        sil_kmeans = silhouette_score(X, kmeans.labels_, metric='euclidean')
    except ValueError:
        pass

    sil_embed = 0.0
    mask = df['cluster_embed'] != -1
    if len(set(df.loc[mask, 'cluster_embed'])) > 1:
        sil_embed = silhouette_score(umap_embeddings[mask], df.loc[mask, 'cluster_embed'])

    def tfidf_diversity_score(dataframe, cluster_col, tfidf_matrix, top_n=10):
        all_top_words = set()
        total_possible = 0

        for cluster_id in dataframe[cluster_col].unique():
            if cluster_id == -1:
                continue

            cluster_indices = np.where(dataframe[cluster_col].values == cluster_id)[0]
            if len(cluster_indices) == 0:
                continue

            # Kiszámoljuk a klaszter átlagos TF-IDF profilját, és vesszük a top_n szót
            mean_tfidf = np.asarray(tfidf_matrix[cluster_indices].mean(axis=0)).ravel()
            top_word_indices = mean_tfidf.argsort()[-top_n:]

            all_top_words.update(top_word_indices)
            total_possible += top_n

        return len(all_top_words) / total_possible if total_possible > 0 else 0



    div_kmeans = tfidf_diversity_score(df, 'cluster', X)
    div_hier = tfidf_diversity_score(sample_df, 'cluster_hier', x_tfidf)
    div_embed = tfidf_diversity_score(df, 'cluster_embed', X) #az elején számoltuk ki, újrafelhasználjuk

    print(f"  Módszer           | Klaszterek száma | Silhouette | Diverzitás")
    print(f"  ------------------|------------------|------------|-----------")
    print(f"  TF-IDF + K-means  | {N_CLUSTERS:>16} | {sil_kmeans:>10.4f} | {div_kmeans:.2f}")
    print(f"  Hierarchikus      | {best_k:>16} | {best_score:>10.4f} | {div_hier:.2f}")
    print(f"  Embedding+HDBSCAN | {n_clusters_embed:>16} | {sil_embed:>10.4f} | {div_embed:.2f}")


    # 7. TÉMAMODELLEZÉS - LDA és NMF
    print("\n" + "=" * 50)
    print("7. LDA és NMF TÉMAMODELLEZÉS (10 téma)...")
    print("=" * 50)

    count_vectorizer = CountVectorizer(max_features=2000, min_df=5, max_df=0.8)
    dtm_count = count_vectorizer.fit_transform(kommentek)

    tfidf_vectorizer_nmf = TfidfVectorizer(max_features=2000, min_df=5, max_df=0.8)
    dtm_tfidf = tfidf_vectorizer_nmf.fit_transform(kommentek)

    feature_names_count = count_vectorizer.get_feature_names_out()
    feature_names_tfidf = tfidf_vectorizer_nmf.get_feature_names_out()

    print("\n[LDA] Modell betanítása...")
    lda = LatentDirichletAllocation(n_components=10, random_state=42, max_iter=20, learning_method='batch')
    lda_doc_topics = lda.fit_transform(dtm_count)

    print("\n[NMF] Modell betanítása...")
    nonzero_rows = dtm_tfidf.getnnz(axis=1) > 0
    dtm_tfidf_filtered = dtm_tfidf[nonzero_rows]
    kommentek_nmf = [kommentek_bert[i] for i in np.where(nonzero_rows)[0]]

    nmf = NMF(n_components=10, random_state=42, l1_ratio=0.5, max_iter=200)
    nmf_doc_topics = nmf.fit_transform(dtm_tfidf_filtered)


    # 7/b. LDA és NMF TÉMÁK ÉS PÉLDÁK KIÍRÁSA
    print("\n[LDA] Témák (top 10 szó és 3 példa):")
    for topic_idx, topic in enumerate(lda.components_):
        top_words_idx = topic.argsort()[-10:][::-1]
        top_words = [feature_names_count[i] for i in top_words_idx]
        print(f"\n🔹 LDA Téma {topic_idx:2d} | Top szavak: {', '.join(top_words)}")

        # Az adott témához leginkább tartozó 3 komment
        top_doc_indices = lda_doc_topics[:, topic_idx].argsort()[-3:][::-1]
        for idx_num, doc_idx in enumerate(top_doc_indices, 1):
            pelda_szoveg = str(kommentek_bert[doc_idx]).replace('\n', ' ').strip()
            if len(pelda_szoveg) > 200:
                pelda_szoveg = pelda_szoveg[:197] + "..."
            print(f"     {idx_num}. példa: \"{pelda_szoveg}\"")

    print("\n[NMF] Témák (top 10 szó és 3 példa):")
    for topic_idx, topic in enumerate(nmf.components_):
        top_words_idx = topic.argsort()[-10:][::-1]
        top_words = [feature_names_tfidf[i] for i in top_words_idx]
        print(f"\n🔹 NMF Téma {topic_idx:2d} | Top szavak: {', '.join(top_words)}")

        # Az adott témához leginkább tartozó 3 komment
        top_doc_indices = nmf_doc_topics[:, topic_idx].argsort()[-3:][::-1]
        for idx_num, doc_idx in enumerate(top_doc_indices, 1):
            pelda_szoveg = str(kommentek_nmf[doc_idx]).replace('\n', ' ').strip()
            if len(pelda_szoveg) > 200:
                pelda_szoveg = pelda_szoveg[:197] + "..."
            print(f"     {idx_num}. példa: \"{pelda_szoveg}\"")

    def plot_top_words(usedmodel, feature_names, n_top_words, title):
        fig, axes = plt.subplots(1, 5, figsize=(16, 5), sharex=False)
        axes = axes.flatten()
        # Az első 5 témát ábrázoljuk
        for topic_idx, topic in enumerate(usedmodel.components_[:5]):
            top_features_ind = topic.argsort()[-n_top_words:]
            top_features = [feature_names[i] for i in top_features_ind]
            weights = topic[top_features_ind]

            ax = axes[topic_idx]
            ax.barh(top_features, weights, height=0.7, color='steelblue' if 'LDA' in title else 'darkorange')
            ax.set_title(f'Téma {topic_idx}', fontdict={'fontsize': 11, 'fontweight': 'bold'})
            ax.tick_params(axis='both', which='major', labelsize=9)
            for i in 'top right'.split():
                ax.spines[i].set_visible(False)
        fig.suptitle(title, fontsize=14, fontweight='bold')
        plt.tight_layout()
        model_name = "LDA" if "LDA" in title else "NMF"
        plt.savefig(os.path.join(results_folder, f"{filename}_5_{model_name}_top_szavak.png"), dpi=300)
        plt.show()

    plot_top_words(lda, feature_names_count, 8, 'LDA - Első 5 téma legfontosabb szavai')
    plot_top_words(nmf, feature_names_tfidf, 8, 'NMF - Első 5 téma legfontosabb szavai')


    # 8. BERTopic
    print("\n" + "=" * 50)
    print("8. BERTopic FUTÁSA...")
    print("=" * 50)

    topic_model = BERTopic(
        language="english",
        calculate_probabilities=False,
        verbose=True,
        n_gram_range=(1, 2),
        min_topic_size=20,
        nr_topics="auto"
    )

    # Újrahasznosítjuk a 4. részben kiszámolt embedding-eket, így sokkal gyorsabb
    topics, probabilities = topic_model.fit_transform(kommentek, embeddings=embeddings)
    topic_info = topic_model.get_topic_info()

    print(f"\nBERTopic által talált témák száma: {len(topic_info) - 1}")
    df['topic_bertopic'] = topics

    # 8/b. BERTopic TÉMÁK ÉS PÉLDÁK KIÍRÁSA
    print("\nBERTopic Témák (top 8 szó és 3 példa):")
    for i in range(1, len(topic_info)):  # A 0. sor a -1 (zaj)
        if i > 15:
            print(f"\n  ... és további {len(topic_info) - 1 - 15} téma.")
            break

        row = topic_info.iloc[i]
        topic_id = int(row['Topic'])
        count = int(row['Count'])
        words = row['Representation']
        print(f"\n🔹 Téma {topic_id:2d} (db: {count:4d}) | Top szavak: {', '.join(words[:8])}") # type: ignore

        # Közvetlenül a topic_info táblázat 'Representative_Docs' oszlopából vesszük a listát
        if 'Representative_Docs' in row.index and isinstance(row['Representative_Docs'], list):
            rep_docs = row['Representative_Docs']
        else:
            rep_docs = df[df['topic_bertopic'] == topic_id]['clean2'].head(3).tolist()

        for idx_num, doc in enumerate(rep_docs[:3], 1): # type: ignore
            pelda_szoveg = str(doc).replace('\n', ' ').strip()
            if len(pelda_szoveg) > 200:
                pelda_szoveg = pelda_szoveg[:197] + "..."
            print(f"     {idx_num}. példa: \"{pelda_szoveg}\"")

    # 1. Top témák és c-TF-IDF súlyaik oszlopdiagramja
    topic_model.visualize_barchart(top_n_topics=10, n_words=8).write_html(os.path.join(results_folder,f"{filename}_6_bertopic_barchart.html"))
    # 2. Témák elhelyezkedése és távolsága 2D térben (Intertopic Distance Map)
    topic_model.visualize_topics().write_html(os.path.join(results_folder,f"{filename}_7_bertopic_intertopic_map.html"))
    # 3. Témák hasonlósági hőtérképe (Heatmap)
    topic_model.visualize_heatmap(top_n_topics=20).write_html(os.path.join(results_folder,f"{filename}_8_bertopic_heatmap.html"))
    print("  > BERTopic interaktív ábrák elmentve HTML fájlokba!")


    # 9. MODELLEK ÉRTÉKELÉSE (Koherencia alapján)
    print("\n" + "=" * 50)
    print("9. KOHERENCIA SZÁMÍTÁS (C_v mutatóval)...")
    print("=" * 50)

    texts = [doc.split() for doc in kommentek]
    dictionary = Dictionary(texts)

    # LDA
    lda_topics = []
    for topic in lda.components_:
        lda_topics.append([feature_names_count[i] for i in topic.argsort()[-20:][::-1]])
    lda_coherence = CoherenceModel(topics=lda_topics, texts=texts, dictionary=dictionary,
                                   coherence='c_v', processes=2).get_coherence()

    # NMF
    nmf_topics = []
    for topic in nmf.components_:
        nmf_topics.append([feature_names_tfidf[i] for i in topic.argsort()[-20:][::-1]])
    nmf_coherence = CoherenceModel(topics=nmf_topics, texts=texts, dictionary=dictionary,
                                   coherence='c_v', processes= 2).get_coherence()

    # BERTopic
    bert_topics = []
    all_topics = topic_model.get_topics()
    for topic_id in sorted(all_topics.keys()):
        if topic_id == -1:  # Zaj klaszter kihagyása
            continue
        bert_topics.append([word for word, _ in all_topics[topic_id][:20]])

    bert_coherence = 0.0
    if len(bert_topics) > 0:
        bert_coherence = CoherenceModel(topics=bert_topics, texts=texts, dictionary=dictionary,
                                        coherence='c_v', processes= 2).get_coherence()

    def topic_diversity(topics_list):
        if not topics_list:
            return 0.0
        # Csak az első 10 szót nézzük minden témában az igazságos összehasonlításért
        top_10_lists = [t[:10] for t in topics_list]
        unique_words = set(word for topic in top_10_lists for word in topic)
        total_words = sum(len(topic) for topic in top_10_lists)
        return len(unique_words) / total_words if total_words > 0 else 0.0

    lda_div = topic_diversity(lda_topics)
    nmf_div = topic_diversity(nmf_topics)
    bert_div = topic_diversity(bert_topics)

    n_bert_topics = len(bert_topics)
    print(f"\n  Modell     | Témák száma | Koherencia (C_v) | Diverzitás")
    print(f"  -----------|-------------|------------------|-----------")
    print(f"  LDA        | {10:<11} | {lda_coherence:>16.4f} | {lda_div:.2f}")
    print(f"  NMF        | {10:<11} | {nmf_coherence:>16.4f} | {nmf_div:.2f}")
    print(f"  BERTopic   | {n_bert_topics:<11} | {bert_coherence:>16.4f} | {bert_div:.2f}")

    # ZÁRÓ VIZUALIZÁCIÓ: Topicmodelling modellek összehasonlítása (Koherencia és Diverzitás alapján)
    models_names = ['LDA', 'NMF', 'BERTopic']
    coherence_vals = [lda_coherence, nmf_coherence, bert_coherence]
    diversity_vals = [lda_div, nmf_div, bert_div]

    x = np.arange(len(models_names))
    width = 0.35

    fig, ax = plt.subplots(figsize=(8, 5))
    rects1 = ax.bar(x - width / 2, coherence_vals, width, label='Koherencia (C_v)', color='royalblue')
    rects2 = ax.bar(x + width / 2, diversity_vals, width, label='Témadiverzitás', color='mediumseagreen')

    ax.set_ylabel('Pontszám (0.0 - 1.0)')
    ax.set_title('Témamodellek teljesítményének összehasonlítása')
    ax.set_xticks(x)
    ax.set_xticklabels(models_names, fontweight='bold')
    ax.set_ylim(0, 1.05)
    ax.legend()
    ax.grid(axis='y', linestyle='--', alpha=0.5)

    # Értékek kiírása az oszlopok tetejére
    ax.bar_label(rects1, fmt='%.3f', padding=3)
    ax.bar_label(rects2, fmt='%.2f', padding=3)

    plt.tight_layout()
    plt.savefig(os.path.join(results_folder, f"{filename}_9_modellek_osszehasonlitasa.png"), dpi=300)
    plt.show()


def askforvideoid():
    filename = input("Kérlek, add meg a videó azonosítóját: ").strip()

    # Ellenőrizzük, hogy létezik-e a fájl, mielőtt visszaadjuk a nevet
    fajl_utvonal = f"tisztitott_adatok/{filename}_tisztazva.csv"
    if os.path.exists(fajl_utvonal):
        return filename
    else:
        print(f"Hiba: Nem található a(z) {fajl_utvonal} fájl!")
        return None


if __name__ == "__main__":
    main()