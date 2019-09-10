import 'package:grinder/grinder.dart';
import 'package:ml_tech/ml_tech.dart' as ml_tech;

Future<void> main(List<String> args) => grind(args);

@Task()
Future<void> start() async {
  ml_tech.analyze();
  await ml_tech.test();
}

@Task()
Future<void> finish() async {
  await ml_tech.uploadCoverage();
}
