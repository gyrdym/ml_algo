/// A type of score to probability mapper function, aka Activation function
///
/// [ScoreToProbMapperType.logit] - a simplest mapper, that maps every
/// input value into a value from interval `[0, 1.0]` using logit.
///
/// Logit is a function, that performs raising exponent into some power
///
/// [ScoreToProbMapperType.softmax] - a mapper, that maps every input value
/// into a value from interval `[0, 1.0]` using a set of logits. (one logit per
/// a class, that a classification algorithm tries to learn)
enum ScoreToProbMapperType {
  logit,
  softmax,
}
