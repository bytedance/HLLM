# Copyright (c) Meta Platforms, Inc. and affiliates.
# Copyright (c) 2024 Bytedance Ltd. and/or its affiliate
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import pandas as pd


def preprocess_interaction(intercation_path, output_path, prefix='books'):
    ratings = pd.read_csv(
        intercation_path,
        sep=",",
        names=["user_id", "item_id", "rating", "timestamp"],
    )
    print(f"{prefix} #data points before filter: {ratings.shape[0]}")
    print(
        f"{prefix} #user before filter: {len(set(ratings['user_id'].values))}"
    )
    print(
        f"{prefix} #item before filter: {len(set(ratings['item_id'].values))}"
    )

    # filter users and items with presence < 5
    item_id_count = (
        ratings["item_id"]
        .value_counts()
        .rename_axis("unique_values")
        .reset_index(name="item_count")
    )
    user_id_count = (
        ratings["user_id"]
        .value_counts()
        .rename_axis("unique_values")
        .reset_index(name="user_count")
    )
    ratings = ratings.join(item_id_count.set_index("unique_values"), on="item_id")
    ratings = ratings.join(user_id_count.set_index("unique_values"), on="user_id")
    ratings = ratings[ratings["item_count"] >= 5]
    ratings = ratings[ratings["user_count"] >= 5]
    ratings = ratings.groupby('user_id').filter(lambda x: len(x['item_id']) >= 5)
    print(f"{prefix} #data points after filter: {ratings.shape[0]}")

    print(
        f"{prefix} #user after filter: {len(set(ratings['user_id'].values))}"
    )
    print(
        f"{prefix} #item ater filter: {len(set(ratings['item_id'].values))}"
    )
    ratings = ratings[['item_id', 'user_id', 'timestamp']]
    ratings.to_csv(output_path, index=False, header=True)


def preprocess_item(item_path, output_path, prefix='books'):
    data = []
    for line in open(item_path):
        json_data = eval(line)
        item_id = json_data.get('asin', '')
        description = json_data.get('description', '')
        title = json_data.get('title', '')

        data.append({
            'item_id': item_id,
            'description': description,
            'title': title
        })

    df = pd.DataFrame(data)
    df.to_csv(output_path, index=False)


if __name__ == '__main__':
    preprocess_interaction("ratings_Books.csv", "amazon_books.csv")
    preprocess_item("meta_Books.json", "amazon_books_item.csv")
