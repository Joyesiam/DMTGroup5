# Task 3: Association Rules -- Product Category Grouping

## The Problem
The standard Apriori algorithm mines association rules at the item level (e.g.,
"customers who buy Pizza Margherita also buy Coca-Cola"). However, when items are
very specific, many potentially interesting patterns are missed because individual
items have low support. Grouping items into higher-level categories (e.g., "Pizza"
instead of "Pizza Margherita" and "Pizza Quattro Formaggi") can reveal broader
patterns that are both more frequent and more actionable.

## Approach: Mining Generalized Association Rules (Srikant & Agrawal, 1995)

Srikant and Agrawal (1995) proposed an extension to Apriori that exploits a
predefined **taxonomy** (concept hierarchy) of items. The key idea is to mine
association rules at multiple levels of the hierarchy simultaneously.

### How It Works

1. **Define a taxonomy tree.** Items are organized into a hierarchy:
   - Level 0 (root): "All Products"
   - Level 1: "Dairy", "Bakery", "Beverages"
   - Level 2: "Milk", "Cheese", "Bread", "Juice"
   - Level 3 (leaf): "Whole Milk", "Skim Milk", "Cheddar", "Gouda"

2. **Extend transactions.** Each transaction is augmented to include all ancestor
   items. If a customer bought "Whole Milk", the transaction also includes "Milk",
   "Dairy", and "All Products".

3. **Apply modified Apriori.** Run the standard Apriori algorithm on the extended
   transactions. This generates candidate itemsets at all levels of the hierarchy.

4. **Prune redundant rules.** A rule like {Dairy} -> {Bread} is redundant if the
   more specific rule {Milk} -> {Bread} has the same support and confidence.
   The algorithm prunes ancestors when their descendants are equally informative.

### Pruning Strategies
- **R-interesting:** A rule involving a generalized item X is pruned if replacing
  X with any of its children yields a rule with support close to X's rule.
- **Minimum improvement threshold:** Only keep the generalized rule if it provides
  a minimum improvement in support over the most specific version.

## Pros and Cons

### Pros
1. **Discovers broader patterns.** Rules like {Pizza} -> {Beer} are found even if
   no single pizza variety has enough support alone. This is especially valuable
   for retailers with large product catalogs.
2. **Reduces the number of rules.** By grouping items, the output is more
   manageable and interpretable for business users. A manager cares about "Dairy"
   trends, not "Gouda 200g" trends.
3. **Handles rare items gracefully.** Items with very low individual frequency
   (e.g., seasonal products) still contribute to patterns through their category.
4. **Actionable insights.** Category-level rules translate more directly to
   business decisions (store layout, promotions, inventory planning).

### Cons
1. **Requires a predefined taxonomy.** The hierarchy must be defined by domain
   experts before mining begins. This is labor-intensive and subjective -- different
   taxonomies yield different rules.
2. **Loss of granularity.** Grouping can mask important item-level patterns. For
   example, {Gouda} -> {Red Wine} might be a strong rule that disappears when
   Gouda is grouped into "Cheese" (because other cheeses do not associate with wine).
3. **Increased computational cost.** Extending transactions with ancestor items
   increases the number of items and candidate itemsets, making the algorithm
   slower. The original paper reports 2-5x runtime increase.
4. **Taxonomy bias.** The results are sensitive to how categories are defined.
   A poorly designed hierarchy can either fragment natural groups or merge
   unrelated items, both degrading rule quality.
5. **Cross-level rules are hard to interpret.** Rules mixing levels (e.g.,
   {Milk} -> {Bakery}) can be confusing: does this mean all bakery products
   or a specific one?
