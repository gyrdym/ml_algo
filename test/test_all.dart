import 'algorithms/kernel_function_factory_test.dart' as kernel_fn_factory_test;
import 'algorithms/kernel_test.dart' as kernel_test;
import 'algorithms/knn_test.dart' as knn_test;
import 'classifier/test_all.dart' as classifier_tests;
import 'common/sequence_elements_distribution_calculator/distribution_calculator_impl_test.dart'
    as sequence_elements_distribution_calculator_impl_test;
import 'cost_function/cost_function_test.dart' as cost_function_test;
import 'cross_validator/cross_validator_impl_test.dart'
    as cross_validator_impl_test;
import 'data_splitter/k_fold_splitter_test.dart' as k_fold_splitter_test;
import 'data_splitter/lpo_splitter_test.dart' as lpo_splitter_test;
import 'helpers/add_intercept.dart' as add_intercept_test;
import 'link_function/link_function_test.dart' as link_function_test;
import 'math/randomizer_test.dart' as randomizer_test;
import 'solver/linear/convergence_detector/convergence_detector_impl_test.dart'
    as convergence_detector_test;
import 'solver/linear/coordinate/coordinate_optimizer_integration_test.dart'
    as coord_optimizer_integration_test;
import 'solver/linear/gradient/gradient_optimizer_integration_test.dart'
    as gradient_optimizer_integration_test;
import 'solver/linear/gradient/gradient_optimizer_test.dart'
    as gradient_optimizer_test;
import 'solver/non_linear/decision_tree/test_all.dart'
  as decision_tree_unit_tests;
import 'regressor/knn_regressor_integration_test.dart'
    as knn_regressor_integration_test;
import 'regressor/knn_regressor_test.dart'
    as knn_regressor_test;

void main() {
  knn_test.main();
  kernel_test.main();
  kernel_fn_factory_test.main();
  classifier_tests.main();
  sequence_elements_distribution_calculator_impl_test.main();
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
  decision_tree_unit_tests.main();
  knn_regressor_integration_test.main();
  knn_regressor_test.main();
  link_function_test.main();
}
