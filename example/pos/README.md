# Point of Sale

How to apply [ml_algo](https://pub.dev/packages/ml_algo) on point of sale data (i.e. tickets/invoices/bills) ?

This first __KDTree__ example provides simple customer segmentation based on purchased items.
Using it a business owner can identify customers who bought __similar items__.
While identifying identical invoices is possible with standard _for_ loops,
Only a KDTree algorithm provides enough flexibility to find __similar items__.

```shell
dart example/pos/kd_contacts.dart
```

## Walkthrough

- The business owner selects a customer (e.g. contactId 35, aka Bobby)
- The program finds Bobby's tickets and vectorize their items 
- The program filters out Bobby's tickets and vectorize all the items
- For each one of Bobby's tickets, the program find the nearest x3 and save them in a list 
- The program selects the top 3 nearest customers based on bought items

## Dataset

- Origin : weebi.com (anonymized)
- Location : West Africa
- Timerange : March 2024 - March 2025

- 59 articles
- 52 contacts (customers)
- 341 tickets