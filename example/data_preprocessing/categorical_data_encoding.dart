import 'package:ml_algo/ml_algo.dart';
import 'package:ml_linalg/range.dart';
import 'package:tuple/tuple.dart';

Future predefinedCategories() async {
  final dataFrame = DataFrame.fromCsv('datasets/black_friday.csv',
    labelName: 'Purchase\r',
    columns: [const Tuple2(2, 3), const Tuple2(5, 7), const Tuple2(11, 11)],
    rows: [const Tuple2(0, 20)],
    categories: {
      'Gender': ['M', 'F'],
      'Age': ['0-17', '18-25', '26-35', '36-45', '46-50', '51-55', '55+'],
      'City_Category': ['A', 'B', 'C'],
      'Stay_In_Current_City_Years': ['0', '1', '2', '3', '4+'],
      'Martial_Status': ['0', '1'],
    },
  );

  final features = await dataFrame.features;
  final genderEncoded = features.submatrix(columns: Range(0, 2));
  final ageEncoded = features.submatrix(columns: Range(2, 9));
  final cityCategiryEncoded = features.submatrix(columns: Range(9, 12));
  final stayInCityEncoded = features.submatrix(columns: Range(12, 16));
  final martialStatusEncoded = features.submatrix(columns: Range(16, 18));

  print('Features:');

  print('feature matrix dimensions: ${features.rowsNum} x '
      '${features.columnsNum};');

  print('==============================');

  print('Gender:');
  print(genderEncoded);

  print('==============================');

  print('Age');
  print(ageEncoded);

  print('==============================');

  print('City category');
  print(cityCategiryEncoded);

  print('==============================');

  print('Stay in current city (years)');
  print(stayInCityEncoded);

  print('==============================');

  print('Martial status');
  print(martialStatusEncoded);
}

Future main() async {
  await predefinedCategories();
}
