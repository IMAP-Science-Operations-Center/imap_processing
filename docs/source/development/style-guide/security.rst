Security
--------

The following items should never be committed in the source code or GitHub issues/pull requests:

* Account credentials of any kind (e.g. database usernames/passwords, AWS credentials, etc.)
* Internal directory structures or filepaths
* Machine names
* Proprietary data

If code needs access to this information, it should be stored in a configuration file that is not part of the
repository.
