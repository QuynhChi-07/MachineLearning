# CustomerSegmentation.py
# Run: python CustomerSegmentation.py

import pandas as pd
import numpy as np
from functools import reduce
import matplotlib.pyplot as plt
from datetime import datetime

from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score

from retail_project.connectors.connector import Connector


# -------------------------
# KẾT NỐI DATABASE
# -------------------------
conn = Connector(database='sakila')
conn.connect()

# Kiểm tra kết nối
sql_test = "SELECT customer_id, first_name, last_name FROM customer LIMIT 5;"
df_test = conn.queryDataset(sql_test)
print("Test connection (sample):")
print(df_test, "\n")


# ====================================================
# 1. PHÂN LOẠI KHÁCH HÀNG THEO FILM
# ====================================================
def customers_by_film(conn):
    sql = """
        SELECT DISTINCT
            f.film_id,
            f.title,
            c.customer_id,
            c.first_name,
            c.last_name
        FROM film f
        JOIN inventory i ON i.film_id = f.film_id
        JOIN rental r ON r.inventory_id = i.inventory_id
        JOIN customer c ON c.customer_id = r.customer_id
        ORDER BY f.film_id, c.customer_id;
    """
    df = conn.queryDataset(sql)
    if df is None or df.empty:
        return pd.DataFrame(), pd.DataFrame()

    df = df.drop_duplicates(subset=['film_id', 'customer_id']).reset_index(drop=True)

    grouped = (
        df.groupby(['film_id', 'title'])[['customer_id', 'first_name', 'last_name']]
        .apply(lambda g: g.drop_duplicates().to_dict('records'))
        .reset_index()
        .rename(columns={0: 'customers'})
    )

    return df, grouped


# ====================================================
# 2. PHÂN LOẠI KHÁCH HÀNG THEO CATEGORY
# ====================================================
def customers_by_category(conn):
    sql = """
        SELECT DISTINCT
            cat.category_id,
            cat.name AS category_name,
            c.customer_id,
            c.first_name,
            c.last_name
        FROM category cat
        JOIN film_category fc ON fc.category_id = cat.category_id
        JOIN film f ON f.film_id = fc.film_id
        JOIN inventory i ON i.film_id = f.film_id
        JOIN rental r ON r.inventory_id = i.inventory_id
        JOIN customer c ON c.customer_id = r.customer_id
        ORDER BY cat.category_id, c.customer_id;
    """
    df = conn.queryDataset(sql)
    if df is None or df.empty:
        return pd.DataFrame(), pd.DataFrame()

    df = df.drop_duplicates(subset=['category_id', 'customer_id']).reset_index(drop=True)

    grouped = (
        df.groupby(['category_id', 'category_name'])[['customer_id', 'first_name', 'last_name']]
        .apply(lambda g: g.drop_duplicates().to_dict('records'))
        .reset_index()
        .rename(columns={0: 'customers'})
    )

    return df, grouped


# ====================================================
# 3. XÂY DỰNG FEATURES KHÁCH HÀNG
# ====================================================
def build_customer_features(conn, reference_date=None):
    sql_rentals = """
        SELECT r.rental_id, r.rental_date, r.return_date, r.inventory_id, r.customer_id, i.film_id
        FROM rental r
        JOIN inventory i ON i.inventory_id = r.inventory_id;
    """
    rentals = conn.queryDataset(sql_rentals)
    if rentals is None or rentals.empty:
        print("Không có dữ liệu thuê phim.")
        return pd.DataFrame()

    rentals['rental_date'] = pd.to_datetime(rentals['rental_date'])
    rentals['return_date'] = pd.to_datetime(rentals['return_date'])

    if reference_date is None:
        reference_date = rentals['rental_date'].max()

    total_rentals = rentals.groupby('customer_id').size().rename('total_rentals').reset_index()
    distinct_films = rentals.groupby('customer_id')['film_id'].nunique().rename('distinct_films').reset_index()

    sql_film_cat = "SELECT film_id, category_id FROM film_category;"
    film_cat = conn.queryDataset(sql_film_cat)

    if film_cat is None or film_cat.empty:
        merged_rentals = rentals.copy()
        merged_rentals['category_id'] = np.nan
        distinct_cats = pd.DataFrame({'customer_id': total_rentals['customer_id'], 'distinct_categories': 0})
    else:
        merged_rentals = rentals.merge(film_cat, on='film_id', how='left')
        distinct_cats = (
            merged_rentals.groupby('customer_id')['category_id']
            .nunique()
            .rename('distinct_categories')
            .reset_index()
        )

    def avg_interval_days(series_dates):
        dates = series_dates.sort_values().values
        if len(dates) < 2:
            return np.nan
        diffs = np.diff(dates).astype('timedelta64[s]') / 86400.0
        return diffs.mean()

    avg_interval = (
        rentals.groupby('customer_id')['rental_date']
        .apply(avg_interval_days)
        .rename('avg_rent_interval_days')
        .reset_index()
    )

    if 'category_id' in merged_rentals.columns:
        cat_counts = merged_rentals.groupby(['customer_id', 'category_id']).size().rename('cnt').reset_index()
        total_by_customer = cat_counts.groupby('customer_id')['cnt'].sum().rename('total_cnt').reset_index()
        cat_counts = cat_counts.merge(total_by_customer, on='customer_id')
        cat_counts['ratio'] = cat_counts['cnt'] / cat_counts['total_cnt']
        favorite_ratio = cat_counts.groupby('customer_id')['ratio'].max().rename('favorite_category_ratio').reset_index()
    else:
        favorite_ratio = pd.DataFrame({'customer_id': total_rentals['customer_id'], 'favorite_category_ratio': 0})

    past90 = reference_date - pd.Timedelta(days=90)
    recent_counts = (
        rentals[rentals['rental_date'] >= past90]
        .groupby('customer_id')
        .size()
        .rename('recent_rentals_90d')
        .reset_index()
    )

    dfs = [total_rentals, distinct_films, distinct_cats, avg_interval, favorite_ratio, recent_counts]
    features = reduce(lambda left, right: pd.merge(left, right, on='customer_id', how='outer'), dfs)

    features['avg_rent_interval_days'] = features['avg_rent_interval_days'].fillna(features['avg_rent_interval_days'].max() or 999)
    num_cols = ['total_rentals', 'distinct_films', 'distinct_categories', 'recent_rentals_90d']
    features[num_cols] = features[num_cols].fillna(0)
    features['favorite_category_ratio'] = features['favorite_category_ratio'].fillna(0)

    return features


# ====================================================
# 4. CHỌN K (ELBOW METHOD)
# ====================================================
def choose_k_elbow(df, k_max=8):
    X = df.copy()
    for col in X.select_dtypes(include=['timedelta']).columns:
        X[col] = X[col].dt.total_seconds() / 86400

    X = X.select_dtypes(include=['number'])
    scaler = StandardScaler()
    Xs = scaler.fit_transform(X)

    distortions = []
    for k in range(1, k_max + 1):
        kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
        kmeans.fit(Xs)
        distortions.append(kmeans.inertia_)

    plt.plot(range(1, k_max + 1), distortions, marker='o')
    plt.title('Elbow Method for KMeans')
    plt.xlabel('Số lượng cụm (k)')
    plt.ylabel('Độ biến dạng (Inertia)')
    plt.show()
    return distortions


# ====================================================
# 5. GOM CỤM KHÁCH HÀNG
# ====================================================
def cluster_customers_kmeans(features_df, k=4):
    if features_df.empty:
        print("Không có dữ liệu đặc trưng để gom cụm.")
        return pd.DataFrame(), None, np.nan

    for col in features_df.select_dtypes(include=['timedelta']).columns:
        features_df[col] = features_df[col].dt.total_seconds() / 86400

    X = features_df.select_dtypes(include=['number'])
    scaler = StandardScaler()
    Xs = scaler.fit_transform(X)

    model = KMeans(n_clusters=k, random_state=42, n_init=10)
    labels = model.fit_predict(Xs)

    features_df = features_df.copy()
    features_df['cluster'] = labels

    sil = silhouette_score(Xs, labels) if len(set(labels)) > 1 else np.nan
    print(f"KMeans: k={k}, silhouette={sil:.3f}")

    return features_df, model, sil


# ====================================================
# 6. CHẠY CHÍNH
# ====================================================
if __name__ == "__main__":
    df_film, _ = customers_by_film(conn)
    df_cat, _ = customers_by_category(conn)
    feat = build_customer_features(conn)

    print("Mẫu đặc trưng khách hàng:")
    print(feat.head(), "\n")

    _ = choose_k_elbow(feat, k_max=8)

    clustered, model, sil = cluster_customers_kmeans(feat, k=3)
    print("Silhouette:", sil, "\n")

    if not clustered.empty:
        summary = clustered.groupby('cluster').agg({
            'customer_id': 'count',
            'total_rentals': 'mean',
            'distinct_films': 'mean',
            'favorite_category_ratio': 'mean'
        })
        print("Tổng quan cụm:")
        print(summary, "\n")


        # ====================================================
        # XUẤT RA WEB (CÓ DROPDOWN LỌC THEO CỤM)
        # ====================================================
        def export_clusters_to_web(df, filename="customer_clusters.html"):
            html_table = df.to_html(index=False, classes="table table-striped table-bordered", border=0)

            # --- Tìm vị trí cột cluster ---
            cluster_col_index = df.columns.get_loc('cluster')

            html_content = f"""
            <html>
            <head>
                <meta charset="utf-8">
                <title>Customer Clusters Report</title>
                <link rel="stylesheet" 
                      href="https://cdn.jsdelivr.net/npm/bootstrap@5.3.0/dist/css/bootstrap.min.css">
                <link rel="stylesheet" 
                      href="https://cdn.datatables.net/1.13.4/css/dataTables.bootstrap5.min.css">
                <script src="https://code.jquery.com/jquery-3.5.1.js"></script>
                <script src="https://cdn.datatables.net/1.13.4/js/jquery.dataTables.min.js"></script>
                <script src="https://cdn.datatables.net/1.13.4/js/dataTables.bootstrap5.min.js"></script>
            </head>
            <body style="padding:20px;">
                <h2 style="text-align:center;">📊 Customer Clusters (KMeans Result, k=3)</h2>
                <hr>

                <div class="mb-3">
                    <label class="form-label"><b>Chọn cụm khách hàng:</b></label>
                    <select id="clusterFilter" class="form-select" style="max-width:300px;">
                        <option value="">Tất cả</option>
                        {''.join([f'<option value="{c}">Cụm {c}</option>' for c in sorted(df["cluster"].unique())])}
                    </select>
                </div>

                <div class="table-responsive">
                    {html_table}
                </div>

                <script>
                $(document).ready(function() {{
                    var table = $('table').DataTable({{
                        pageLength: 20,
                        lengthMenu: [10, 20, 50, 100],
                        order: [[ {cluster_col_index}, 'asc' ]]
                    }});

                    $('#clusterFilter').on('change', function() {{
                        var val = $(this).val();
                        table.column({cluster_col_index}).search(val).draw();
                    }});
                }});
                </script>
            </body>
            </html>
            """

            with open(filename, "w", encoding="utf-8") as f:
                f.write(html_content)
            print(f"Đã xuất kết quả ra file web: {filename}")


        # Gọi hàm xuất
        export_clusters_to_web(clustered, "customer_clusters.html")

