# Elo7 Data Science Challenge

This challenge was originally part of the technical assessment for a Data Scientist position at Elo7. Its objective is to evaluate how candidates approach the development of a complete machine learning solution to a problem similar to those encountered by the company's Data Science team.

## Problem Description

At Elo7, we work with a catalog of approximately six million unique products. All information displayed on a product page is provided by the seller, resulting in largely unstructured textual data and numerical attributes that may not always reflect realistic market values. As a consequence, every product is effectively unique, making it challenging to identify meaningful patterns.

Every day, customers interact with our search engine with two primary goals: to find a specific product they already have in mind or to discover new products that match their interests.

A search query is one of the strongest signals of a user's intent. Additional information becomes available after users interact with products, such as viewing or purchasing them. The question we would like you to explore is:

> **Can user intent be inferred solely from the search query?**

To investigate this question, use the provided dataset to accomplish the following objectives:

- Develop a **supervised** model to classify products into categories using the labeled `category` field.
- Develop an **unsupervised** system that groups search queries into at least two classes representing different types of user intent.
- Develop a **hybrid** system that combines the previous two components. Given an arbitrary search query, the system should:
  - Predict the user's likely intent.
  - Predict the product category the query would belong to if it represented a product.
  - Recommend the ten most relevant products based on both the search query and the inferred user intent.

The following sections describe these objectives in greater detail and explain how your solution will be evaluated.

## Dataset

The dataset used in this challenge consists of a sample of data from Elo7. It can be downloaded using the following link:

https://elo7-datasets.s3.amazonaws.com/data_scientist_position/elo7_recruitment_dataset.csv

If you experience any issues accessing the dataset, please contact us and we will be happy to help.

In summary, the dataset contains **38,507 records** distributed across six categories:

- `Bebê` (*Baby*)
- `Bijuterias e Jóias` (*Jewelry & Accessories*)
- `Decoração` (*Home Decor*)
- `Lembrancinhas` (*Party Favors*)
- `Papel e Cia` (*Paper Crafts*)
- `Outros` (*Others*)

Each record represents a user click on a product resulting from a search query.

The dataset contains the following columns:

- `product_id` – Product identifier.
- `seller_id` – Seller identifier.
- `query` – Search query entered by the user.
- `search_page` – Search results page where the product appeared (minimum: 1, maximum: 5).
- `position` – Position of the product within the search results page (minimum: 0, maximum: 38).
- `title` – Product title.
- `concatenated_tags` – Product tags provided by the seller, concatenated into a single space-separated string.
- `creation_date` – Date the product was created on the Elo7 platform.
- `price` – Product price in Brazilian reais (BRL).
- `weight` – Product weight in grams, as reported by the seller.
- `express_delivery` – Indicates whether the product is available for immediate shipping (`1`) or not (`0`).
- `minimum_quantity` – Minimum quantity required for purchase.
- `view_counts` – Number of product views during the previous three months.
- `order_counts` – Number of purchases during the previous three months.
- `category` – Product category.

## Solution Development

Your solution should be structured according to the following stages:

1. **Exploratory Data Analysis**
2. **Product Classification System**
3. **Search Intent System**
4. **System Evaluation**
5. **System Integration**
6. **Solution Delivery**

The following sections describe each stage in more detail and explain the criteria that will be used to evaluate your solution.

## Part 1 - Exploratory Data Analysis

This stage is one of the most important aspects of any data science project. Before applying any machine learning algorithm, it is essential to first understand the problem and the data. We therefore expect you to perform an exploratory data analysis (EDA) and identify the most relevant characteristics of the dataset. You are free to choose any tools, techniques, or algorithms that you consider appropriate for this stage.

To help organize your analysis, we recommend explicitly stating the questions you intend to answer. This will provide a clearer structure and make your investigation easier to follow.

Be creative throughout your exploration! The insights you uncover during this stage will likely inspire ideas for the product classification system developed in the next part of the challenge.

## Part 2 - Product Classification System

Develop a proof-of-concept classifier capable of assigning products to their respective categories. Your model should use at least one textual feature and one numerical feature as inputs.

We intentionally do not prescribe any specific algorithm or methodology, as doing so would bias your solution. Instead, we expect you to document every important decision made throughout the development process, including approaches that did not produce satisfactory results.

Do not focus solely on achieving the highest possible predictive performance. Instead, we are primarily interested in understanding your reasoning and methodology. Build a compelling narrative around the data by combining code with clear explanations of your thought process and the rationale behind your decisions.

## Part 3 - Search Intent System

Develop a classifier that categorizes **search queries** into user intent classes. Notice that the dataset does **not** contain labels for user intent. It is your responsibility to define these intent classes by investigating the data and developing an appropriate modeling strategy.

To help you get started, here are a few examples of possible intent classes:

- **Two classes:** users who are exploring new products versus users who already know exactly what they want.
- **Two or more classes:** users searching for products within different price ranges (e.g., up to R$25.00, between R$25.00 and R$100.00, and above R$100.00).
- **Two or more classes:** users searching for products within specific categories versus users with broader or more general interests.

Since no ground-truth intent labels are provided, this is primarily an **unsupervised learning** task. However, you are free to devise your own labeling strategy and formulate it as a supervised learning problem if you believe that approach is appropriate. In either case, be sure to justify your methodology.

Your system should ultimately accept a search query as input and return the corresponding user intent class.

The primary evaluation criteria for this stage are:

- The methodology used to define the intent classes.
- The labeling strategy, if one is employed.
- The implementation and quality of the resulting classifier.

## Part 4 - System Evaluation

In this stage, you should define appropriate evaluation metrics for the systems developed in Parts 2 and 3.

Selecting suitable evaluation metrics is a fundamental aspect of machine learning. Different metrics are appropriate for different problems, and there is rarely a single metric that is universally applicable. Consider which evaluation techniques are most appropriate for your proposed solutions and justify your choices.

Focus on explaining **why** the selected metrics are appropriate rather than spending time optimizing your models to achieve the highest possible scores. We are primarily interested in understanding how you evaluate machine learning systems.

> **Note:** Once again, be **creative**!

## Part 5 – System Integration

Now that the individual systems are working, the next step is to perform one of the most important tasks in a data scientist’s workflow: integrating multiple models into a unified system.

An interesting direction is to make the recommendation system compatible with the user intent classes previously defined. Another possible application is to use the combined system to generate product recommendations directly from a search query, returning the ten most relevant products for the user.

You have full freedom to design how these components are integrated into a single system, as long as the required functionality of the final solution is respected.

The final system must:

- Accept any search query as input (not limited to queries present in the dataset).
- Produce three outputs:
  1. The predicted product category for the query, as if it were a product description. Note that in this case no numerical features are available; you should explore how to handle this limitation and clearly document your approach.
  2. The predicted user intent class.
  3. A list of the ten most relevant recommended products, including their `id` and `title`.

To help guide your design, here are some possible recommendation strategies:

- Recommend products similar to what the user is searching for.
- Recommend complementary products to the user’s query.
- Recommend popular products, based on overall views or purchase counts.

Another interesting perspective is to consider what happens when product classification is applied to search queries. This can be used to estimate user demand patterns. For example, even if the catalog contains many products in the “Home Decor” category, the majority of search volume might correspond to “Party Favors.” Such insights can be used to inform sellers about market demand and guide their product strategy.

The main evaluation criteria for this stage are:

- Creativity in the design of the integrated system.
- Robustness of the integration between components.
- Quality and methodology of the recommendation approach.

## Part 6 – System Delivery

Finally, you must deliver all investigation, experimentation, reasoning, and implementation of your solution in one or more Jupyter Notebooks, as well as a Python script capable of executing all systems as shown in the examples below.

### 1. Product Classification

```bash
$ teste_ds.py --category "{'feature_1':<feature_1_input>,'feature_2':<feature_2_input>,...}"
>>> "<category_name>"
```

### 2. Search Intent Classification

```bash
$ teste_ds.py --intent "<user_search_query>"
>>> "<intent_class_name>"
```

### 3. Hybrid System

```bash
$ teste_ds.py --recommendation "<user_search_query>"
>>> "<category_name>"
>>> "<intent_class_name>"
>>> "<product_id_1>,<recommended_product_title_1>"
>>> "<product_id_2>,<recommended_product_title_2>"
...
>>> "<product_id_10>,<recommended_product_title_10>"
```

### General Notes

You must submit one or more Jupyter Notebooks containing your full solution. Be sure to document your code thoroughly and use Markdown cells to explain your approach in detail. Clearly describe your reasoning and justify all methodological choices. Explicitly state which algorithms were used and what preprocessing steps were applied, along with the motivation behind each decision.

You may refer to the list of supported Jupyter kernels here:
https://github.com/jupyter/jupyter/wiki/Jupyter-kernels

and installation instructions here:
https://ipython.readthedocs.io/en/latest/install/kernel_install.html

Upload the final `.ipynb` files, a `README.md`, and a `requirements.txt` file (generated using `pip freeze > requirements.txt`) to your personal GitHub repository. Make the repository public and share the link via email.

If you use a language other than Python, clearly explain how to run your project locally. If you choose Python, please submit using Python 3+.

You are free to use libraries such as scikit-learn and SciPy. However, you must be able to explain why a given algorithm is appropriate for a given task.

There is no single correct solution. We are primarily interested in understanding your reasoning process.

---

### Important

Be creative in your solution. The work of a data scientist involves technical knowledge, scientific methodology, and creativity when solving complex problems.

Try to formulate hypotheses, build a product classification algorithm, and define evaluation metrics that test those hypotheses.

Good luck, and feel free to contact us if you have any questions.