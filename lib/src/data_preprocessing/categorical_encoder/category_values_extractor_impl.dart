import 'package:ml_algo/src/data_preprocessing/categorical_encoder/category_values_extractor.dart';

class CategoryValuesExtractorImpl<T> implements CategoryValuesExtractor<T> {
  const CategoryValuesExtractorImpl();

  @override
  List<T> extractCategoryValues(List<T> values) {
    final unique = <T, bool>{};
    for (int i = 0; i < values.length; i++) {
      unique.putIfAbsent(values[i], () => true);
    }
    return unique.keys.toList(growable: false);
  }
}
