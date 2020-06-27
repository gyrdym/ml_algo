import 'package:process_run/shell.dart';

Future runE2ETests() => Shell().run(
'''

echo Running e2e tests...

pub run test e2e -p vm

echo e2e tests finished

''');

void main() async {
  await runE2ETests();
}
