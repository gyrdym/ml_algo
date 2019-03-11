/// Types of categorical data encoders
///
/// [CategoricalDataEncoderType.oneHot] One-hot encoder. Encodes every
/// categorical value to a list of length, that is equal to the number of all
/// possible category's values. Each element of the list is a binary value: `1`
/// for the current value, `0` - for the rest values.
///
/// For example:
///
/// Category `'AGE'` given. Its possible values:
/// ```
/// ['0-17', '18-30', '31+']
/// ```
///
/// '0-17' will be encoded as [1.0, 0.0, 0.0]
///
/// '18-30' will be encoded as [0.0, 1.0, 0.0]
///
/// '31+' will be encoded as [0.0, 0.0, 1.0]
///
/// Let's say, we have some data of this category - a list of `'AGE'` values:
/// ```
/// ['0-17', '0-17', '18-30', '18-30', '18-30', '31+']
/// ```
///
/// After one-hot encoding the data will look like:
/// ```
/// [[1.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 1.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]]
/// ```
///
/// [CategoricalDataEncoderType.ordinal] Ordinal encoder. Encodes every
/// categorical value to an ordinal number.
///
enum CategoricalDataEncoderType {
  ordinal,
  oneHot,
}
