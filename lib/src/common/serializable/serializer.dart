abstract class Serializer<T> {
  /// Returns a serialized object
  Map<String, dynamic> serialize(T model);

  /// Returns a deserialized object
  T deserialize(Map<String, dynamic> serializedModel);
}
