/// Types of categorical data encoders
/// [CategoricalDataEncoderType.oneHot] One-hot encoder. Encodes every categorical value to a sequence of all possible
/// values of its category: 1 for the given value, 0 - for the rest values. For example:
/// Category 'GENDER' given. Its possible values: 'female', 'male'. Also, we have some data to encode - a list of
/// 'GENDER' values: 'female', 'female', 'male', 'male', 'male', 'female'. After one-hot encoding the data will be as:
/// [[1, 0], [1, 0], [0, 1], [0, 1], [0, 1], [1, 0]]
/// [CategoricalDataEncoderType.ordinal] Ordinal encoder. Encodes every categorical value to a ordinal number
enum CategoricalDataEncoderType {
  ordinal,
  oneHot,
}
