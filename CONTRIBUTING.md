# Contributing

Feel free to create a pull request or an issue, your attempt to make a contribution won't be unnoticed 
and it will be processed as soon as possible.

Before changing the code of the project, please do the following steps:

- Update git submodules via `git submodule update --init --recursive` command
- Update all the dart dependencies via `pub get` command

After making changes, please do the following steps:
- Navigate to the root of the library and run `./build.sh`. The command might produce changes for `.g.dart` files
- Add unit tests for a new functionality
- Update a version in the pubspec.yaml, to do so please use [semver](https://semver.org/)
- Add a record with the version and a short description of the change you made to CHANGELOG.md
- Add/change info in README.md 
