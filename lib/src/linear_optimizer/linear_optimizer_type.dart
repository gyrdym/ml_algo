/// Linear optimization types
///
/// The whole process of learning linear models confines to just finding
/// coefficients of features of predicting line/hyperplane. There are several
/// algorithms to find the coefficients that minimize a cost function best.
enum LinearOptimizerType {
  /// Original gradient descent/ascent optimization, only L2 regularization is
  /// applicable while optimizing a function using this method
  gradient,

  /// Original coordinate descent optimization, only L1 regularization is
  /// applicable while optimizing a function using this method
  coordinate,

  /// Closed form solution for finding coefficients, based on feature matrix
  /// inversion
  closedForm,
}
