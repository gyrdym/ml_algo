abstract class Serializer<T> {
  /// Returns a serialized object
  Map<String, dynamic> serialize(T model);

  /// Returns an original object
  T deserialize(Map<String, dynamic> serializedModel);
}
