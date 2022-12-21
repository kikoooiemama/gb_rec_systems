import pandas as pd
import numpy as np

# Для работы с матрицами
from scipy.sparse import csr_matrix

# Матричная факторизация
from implicit.als import AlternatingLeastSquares
from implicit.nearest_neighbours import ItemItemRecommender  # нужен для одного трюка
from implicit.nearest_neighbours import bm25_weight, tfidf_weight


class MainRecommender:
    """Рекоммендации, которые можно получить из ALS

    Input
    -----
    user_item_matrix: pd.DataFrame
        Матрица взаимодействий user-item
    """

    def __init__(self, data, weighting=True):
        self.user_item_matrix = self.prepare_matrix(data)  # pd.DataFrame
        self.id_to_itemid, self.id_to_userid, self.itemid_to_id, self.userid_to_id = self.prepare_dicts(
            self.user_item_matrix)

        if weighting:
            self.user_item_matrix = bm25_weight(self.user_item_matrix.T).T

        self.model = self.fit(self.user_item_matrix)
        self.own_recommender = self.fit_own_recommender(self.user_item_matrix)

    @staticmethod
    def prepare_matrix(data):
        """Подготовка разреженной матрицы"""

        user_item_matrix = pd.pivot_table(data,
                                          index='user_id',
                                          columns='item_id',
                                          values='quantity',
                                          aggfunc='count',
                                          fill_value=0
                                          )

        user_item_matrix = user_item_matrix.astype(float)

        return user_item_matrix

    @staticmethod
    def prepare_dicts(user_item_matrix):
        """Подготавливает вспомогательные словари"""

        userids = user_item_matrix.index.values
        itemids = user_item_matrix.columns.values

        matrix_userids = np.arange(len(userids))
        matrix_itemids = np.arange(len(itemids))

        id_to_itemid = dict(zip(matrix_itemids, itemids))
        id_to_userid = dict(zip(matrix_userids, userids))

        itemid_to_id = dict(zip(itemids, matrix_itemids))
        userid_to_id = dict(zip(userids, matrix_userids))

        return id_to_itemid, id_to_userid, itemid_to_id, userid_to_id

    @staticmethod
    def fit_own_recommender(user_item_matrix):
        """Обучает модель, которая рекомендует товары, среди товаров, купленных юзером"""

        own_recommender = ItemItemRecommender(K=1, num_threads=4)
        own_recommender.fit(csr_matrix(user_item_matrix).tocsr())

        return own_recommender

    @staticmethod
    def fit(user_item_matrix, n_factors=20, regularization=0.001, iterations=15, num_threads=4):
        """Обучает ALS"""

        model = AlternatingLeastSquares(factors=n_factors,
                                        regularization=regularization,
                                        iterations=iterations,
                                        num_threads=num_threads)

        model.fit(csr_matrix(user_item_matrix).tocsr())

        return model

    def _get_similar_item(self, item_id):
        """Находит товар, похожий на item_id"""

        recs = self.model.similar_items(self.itemid_to_id[item_id], N=2)
        top_rec = recs[0][1]
        return self.id_to_itemid[top_rec]

    def get_similar_items_recommendation(self, user, N=5):
        """Рекомендуем товары, похожие на топ-N купленных юзером товаров"""

        # выясняем топ-N купленных юзером товаров
        top_users_purchases = self.user_item_matrix.iloc[self.userid_to_id[user]].sort_values(ascending=False).head(
            N).reset_index()
        # находим по одной рекомендации для каждого топ-N товара.
        res = top_users_purchases['item_id'].apply(lambda x: self._get_similar_item(x)).tolist()
        res = np.array(res)

        assert len(res) == N, 'Количество рекомендаций != {}'.format(N)
        return res

    def _get_recommendations(self, user, model, N=5):
        res = [self.id_to_itemid[rec] for rec in
               model.recommend(userid=self.userid_to_id[user],
                               user_items=csr_matrix(self.user_item_matrix).tocsr(),
                               N=N,
                               filter_already_liked_items=False,
                               filter_items=None,
                               recalculate_user=True)[0]]
        return res

    def get_similar_users_recommendation(self, user, N=5):
        """Рекомендуем топ-N товаров, среди купленных похожими юзерами"""

        res = []
        # Находим похожих юзеров
        similar_users = self.model.similar_users(self.userid_to_id[user], N=6)
        similar_users = [rec for rec in similar_users[0]][1:]
        # Находим купленные топ товары, купленные этими юзерами
        for user in similar_users:
            user_rec = self._get_recommendations(user, model=self.own_recommender, N=1)
            res.append(user_rec)
        res = np.array(res).flatten()

        assert len(res) == N, 'Количество рекомендаций != {}'.format(N)
        return res