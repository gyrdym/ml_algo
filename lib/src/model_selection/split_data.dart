import 'package:ml_dataframe/ml_dataframe.dart';

List<DataFrame> splitData(DataFrame data, Iterable<num> ratios) {
  if (ratios.isEmpty) {
    throw Exception('Ratio collection must contain at least one element');
  }

  final inputRows = data
      .rows
      .toList();
  var start = 0;

  return ratios
      .map((ratio) {
        if (ratio <= 0 || ratio >= 1) {
          throw Exception('Ratio value must be within range 0..1 (both '
              'exclusive), $ratio given');
        }

        final end = start + (inputRows.length * ratio).ceil();
        final rows = inputRows.sublist(start, end);

        start = end;

        return DataFrame(rows, headerExists: false, header: data.header);
      })
      .toList()
      .followedBy([
        DataFrame(
          inputRows.sublist(start),
          headerExists: false,
          header: data.header
        ),
      ])
      .toList(growable: false);
}
