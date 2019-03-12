import 'package:ml_algo/src/data_preprocessing/categorical_encoder/category_values_extractor.dart';

class CategoryValuesExtractorImpl implements CategoryValuesExtractor {
  const CategoryValuesExtractorImpl();

  @override
  List<String> extractCategoryValues(List<String> values) {
    final unique = <String, bool>{};
    for (int i = 0; i < values.length; i++) {
      unique.putIfAbsent(values[i], () => true);
    }
    return unique.keys.toList(growable: false);
  }
}
