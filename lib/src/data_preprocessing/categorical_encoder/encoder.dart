/// A categorical data encoder. Contains names and values of the categories that supposed to be encoded and provides
/// method for data encoding
abstract class CategoricalDataEncoder {
  /// Target categories. The key of the map - a category name, the value - a collection of all possible category values
  Map<String, Iterable<Object>> get categories;

  /// Encodes passed categorical value to numerical representation
  Iterable<double> encode(String categoryLabel, Object value);
}
