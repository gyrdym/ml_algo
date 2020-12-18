import 'package:json_annotation/json_annotation.dart';
import 'package:ml_algo/src/knn_solver/knn_solver.dart';
import 'package:ml_algo/src/knn_solver/knn_solver_impl.dart';

class KnnSolverJsonConverter implements
    JsonConverter<KnnSolver, Map<String, dynamic>> {
  const KnnSolverJsonConverter();

  @override
  KnnSolver fromJson(Map<String, dynamic> json) =>
      KnnSolverImpl.fromJson(json);

  @override
  Map<String, dynamic> toJson(KnnSolver solver) => solver.toJson();
}
