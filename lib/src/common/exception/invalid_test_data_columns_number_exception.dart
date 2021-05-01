class InvalidTestDataColumnsNumberException implements Exception {
  InvalidTestDataColumnsNumberException(int expected, int received)
      : message = 'Unexpected columns number in test data, '
            'expected $expected, received $received';

  final String message;

  @override
  String toString() => message;
}
