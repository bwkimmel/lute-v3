"""
Database migration manager.

Runs migrations and logs them to a _migrations table so they aren't
run twice.  Runs repeatable migrations always.
"""

import os

class SqliteMigrator:
    """
    Sqlite migrator class.

    Follows the principles documented in
    https://github.com/jzohrab/DbMigrator/blob/master/docs/managing_database_changes.md
    """

    def __init__(self, location, repeatable, showlogging=False):
        self.location = location
        self.repeatable = repeatable
        self.showlogging = showlogging

    def has_migrations(self, conn):
        """
        Return True if have non-applied migrations.
        """
        outstanding = self._get_pending(conn)
        return len(outstanding) > 0

    def _get_pending(self, conn):
        """
        Get all non-applied (one-time) migrations.
        """
        allfiles = []
        os.chdir(self.location)
        allfiles = [s for s in os.listdir() if s.endswith('.sql')]
        allfiles.sort()
        outstanding = [f for f in allfiles if self._should_apply(conn, f)]
        return outstanding

    def do_migration(self, conn):
        """
        Run all migrations, then all repeatable migrations.
        """
        self._process_folder(conn)
        self._process_repeatable(conn)

    def _log(self, message):
        """
        Hacky debug logging.
        """
        if self.showlogging:
            print(message)

    def _process_folder(self, conn):
        """
        Run all pending migrations.  Write executed script to
        _migrations table.
        """
        outstanding = self._get_pending(conn)
        self._log(f"running {len(outstanding)} migrations in {self.location}")
        for f in outstanding:
            try:
                self._process_file(conn, f)
            except Exception as e:
                msg = str(e)
                print(f"\nFile {f} exception:\n{msg}\n")
                raise e
            self._add_migration_to_database(conn, f)

    def _process_repeatable(self, conn):
        """
        Run all repeatable migrations.
        """
        folder = self.repeatable
        os.chdir(folder)
        files = [f for f in os.listdir() if f.endswith('.sql')]
        self._log(f"running {len(files)} repeatable migrations in {folder}")
        for f in files:
            try:
                self._process_file(conn, f, False)
            except Exception as e:
                msg = str(e)
                print(f"\nFile {f} exception:\n{msg}\n")
                raise e

    def _should_apply(self, conn, filename):
        """
        True if a migration hasn't been run yet.
        """
        if os.path.isdir(filename):
            return False
        sql = f"select count(filename) from _migrations where filename = '{filename}'"
        res = conn.execute(sql).fetchone()
        return res[0] == 0

    def _add_migration_to_database(self, conn, filename):
        """
        Track the executed migration in _migrations.
        """
        self._log(f'  tracking migration {filename}')
        conn.execute('begin transaction')
        conn.execute(f"INSERT INTO _migrations values ('{filename}')")
        conn.execute('commit transaction')

    def _process_file(self, conn, f, showmsg=True):
        """
        Run the given file.
        """
        if showmsg:
            self._log(f"  running {f}")
        with open(f, 'r', encoding='utf8') as sql_file:
            commands = sql_file.read()
            self._exec_commands(conn, commands)

    def _exec_commands(self, conn, sql):
        """
        Execute all commands in the given file.
        """
        commands = [ c.strip() for c in sql.split(';') ]
        commands = [ c for c in commands if c != '' ]
        try:
            conn.execute('begin transaction')
            for c in commands:
                conn.execute(c)
            conn.execute('commit transaction')
        except Exception as e:
            conn.execute('rollback')
            raise e