import 'dart:async';

import 'package:ml_linalg/matrix.dart';
import 'package:ml_linalg/vector.dart';

/// A wrapper for raw data (e.g., csv-data, sql-data or some other format). Contains features and labels extracted from
/// raw data
abstract class MLData<E> {
  /// A data structure, containing just dataset column headers (generally, first row of a dataset). It may be omitted
  /// (in this case `null` will be returned)
  Future<List<String>> get header;

  /// Processed and ready to use (by machine learning algorithms) dataset features. Keep in mind, that the number of
  /// columns of the feature matrix may differ from the number of elements in [header] because of categorical data,
  /// that might present in the source dataset
  Future<MLMatrix<E>> get features;

  /// Processed and ready to use (by machine learning algorithms) dataset labels (Target values, e.g. class labels or
  /// regression values)
  Future<MLVector<E>> get labels;
}