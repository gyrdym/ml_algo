import 'dart:io';
import 'dart:convert';

void main() {
  Process.start('pub run test', [], runInShell: true).then((Process process) {
    process.stdout.transform(UTF8.decoder).listen((data) {stdout.write(data);});
    process.stderr.transform(UTF8.decoder).listen((data) {stderr.write(data);});
  });
}
