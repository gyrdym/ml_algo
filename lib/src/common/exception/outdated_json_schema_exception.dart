class OutdatedJsonSchemaException implements Exception {
  @override
  String toString() => 'The model is outdated, some hyperparameters are absent, '
      'thus it\'s impossible to retrain it. Please generate a new model.';
}
