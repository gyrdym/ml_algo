import 'package:ml_algo/src/tree_trainer/leaf_label/leaf_label.dart';

TreeLeafLabel? fromLeafLabelJson(Map<String, dynamic>? json) =>
    json != null
        ? TreeLeafLabel.fromJson(json)
        : null;
