import 'algorithms/knn_test.dart' as knn_test;
import 'algorithms/kernel_test.dart' as kernel_test;
import 'algorithms/kernel_function_factory_test.dart' as kernel_fn_factory_test;
import 'classifier/logistic_regressor_integration_test.dart'
    as logistic_regressor_integration_test;
import 'classifier/logistic_regressor_test.dart' as logistic_regressor_test;
import 'classifier/softmax_regressor_test.dart' as softmax_regressor_test;
import 'cost_function/cost_function_test.dart' as cost_function_test;
import 'data_preprocessing/intercept_preprocessor_test.dart'
    as intercept_preprocessor_test;
import 'data_splitter/data_splitter_test.dart' as data_splitter_test;
import 'math/randomizer_test.dart' as randomizer_test;
import 'optimizer/convergence_detector/convergence_detector_impl_test.dart'
    as convergence_detector_test;
import 'optimizer/coordinate/coordinate_optimizer_integration_test.dart'
    as coord_optimizer_integration_test;
import 'optimizer/gradient/gradient_optimizer_integration_test.dart'
    as gradient_optimizer_integration_test;
import 'optimizer/gradient/gradient_optimizer_test.dart'
    as gradient_optimizer_test;
import 'regressor/knn_regressor_integration_test.dart'
    as knn_regressor_integration_test;
import 'regressor/knn_regressor_test.dart'
    as knn_regressor_test;
import 'score_to_prob_mapper/score_to_prob_mapper_test.dart'
    as score_to_prob_mapper_test;

void main() {
  knn_test.main();
  kernel_test.main();
  kernel_fn_factory_test.main();
  logistic_regressor_integration_test.main();
  logistic_regressor_test.main();
  softmax_regressor_test.main();
  cost_function_test.main();
  intercept_preprocessor_test.main();
  data_splitter_test.main();
  randomizer_test.main();
  convergence_detector_test.main();
  coord_optimizer_integration_test.main();
  gradient_optimizer_integration_test.main();
  gradient_optimizer_test.main();
  knn_regressor_integration_test.main();
  knn_regressor_test.main();
  score_to_prob_mapper_test.main();
}
