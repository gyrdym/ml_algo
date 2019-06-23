import 'algorithms/kernel_function_factory_test.dart' as kernel_fn_factory_test;
import 'algorithms/kernel_test.dart' as kernel_test;
import 'algorithms/knn_test.dart' as knn_test;
import 'classifier/logistic_regressor_integration_test.dart'
    as logistic_regressor_integration_test;
import 'classifier/logistic_regressor_test.dart' as logistic_regressor_test;
import 'classifier/softmax_regressor_test.dart' as softmax_regressor_test;
import 'cost_function/cost_function_test.dart' as cost_function_test;
import 'cross_validator/cross_validator_impl_test.dart'
  as cross_validator_impl_test;
import 'data_splitter/k_fold_splitter_test.dart' as k_fold_splitter_test;
import 'data_splitter/lpo_splitter_test.dart' as lpo_splitter_test;
import 'helpers/add_intercept.dart' as add_intercept_test;
import 'link_function/link_function_test.dart' as link_function_test;
import 'math/randomizer_test.dart' as randomizer_test;
import 'optimizer/linear/convergence_detector/convergence_detector_impl_test.dart'
    as convergence_detector_test;
import 'optimizer/linear/coordinate/coordinate_optimizer_integration_test.dart'
    as coord_optimizer_integration_test;
import 'optimizer/linear/gradient/gradient_optimizer_integration_test.dart'
    as gradient_optimizer_integration_test;
import 'optimizer/linear/gradient/gradient_optimizer_test.dart'
    as gradient_optimizer_test;
import 'optimizer/non_linear/decision_tree/assessor/classifier_stump_assesor_test.dart'
    as classifier_stump_assesor_test;
import 'optimizer/non_linear/decision_tree/stump_selector/number_based/node_splitter/node_splitter_impl_test.dart'
    as node_splitter_impl_test;
import 'optimizer/non_linear/decision_tree/stump_selector/number_based/greedy_number_based_stump_selector_test.dart'
    as greedy_number_based_stump_selector_test;
import 'optimizer/non_linear/decision_tree/leaf_detector/leaf_detector_impl_test.dart'
    as leaf_detector_impl_test;
import 'regressor/knn_regressor_integration_test.dart'
    as knn_regressor_integration_test;
import 'regressor/knn_regressor_test.dart'
    as knn_regressor_test;

void main() {
  knn_test.main();
  kernel_test.main();
  kernel_fn_factory_test.main();
  logistic_regressor_integration_test.main();
  logistic_regressor_test.main();
  softmax_regressor_test.main();
  cost_function_test.main();
  cross_validator_impl_test.main();
  k_fold_splitter_test.main();
  lpo_splitter_test.main();
  add_intercept_test.main();
  randomizer_test.main();
  convergence_detector_test.main();
  coord_optimizer_integration_test.main();
  gradient_optimizer_integration_test.main();
  gradient_optimizer_test.main();
  classifier_stump_assesor_test.main();
  leaf_detector_impl_test.main();
  node_splitter_impl_test.main();
  greedy_number_based_stump_selector_test.main();
  knn_regressor_integration_test.main();
  knn_regressor_test.main();
  link_function_test.main();
}
