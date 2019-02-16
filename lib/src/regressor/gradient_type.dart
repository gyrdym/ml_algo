/// A type of gradient descent optimizer
///
/// [GradientType.stochastic] uses just one row of the entire dataset per one
/// fitting iteration
///
/// [GradientType.miniBatch] uses several rows of the dataset per one fitting
/// iteration
///
/// [GradientType.batch] uses all dataset per one fitting iteration
///
enum GradientType { stochastic, miniBatch, batch }
