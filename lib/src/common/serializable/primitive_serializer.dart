abstract class PrimitiveSerializer<T> {
  /// Returns a serialized object
  String serialize(T value);

  /// Returns an original value
  T deserialize(String serializedValue);
}
