import io
import re
import tokenize


def strip_jcg_comments(code: str) -> str:
    result = list(code)
    i = 0
    n = len(code)

    in_single_line_comment = False
    in_multi_line_comment = False
    string_delimiter = None
    in_verbatim_string = False

    while i < n:
        c = code[i]
        next_c = code[i + 1] if i + 1 < n else ''

        if in_single_line_comment:
            if c != '\n':
                result[i] = ' '
            else:
                in_single_line_comment = False
            i += 1
            continue

        if in_multi_line_comment:
            if c == '*' and next_c == '/':
                result[i] = result[i + 1] = ' '
                in_multi_line_comment = False
                i += 2
            else:
                if c != '\n':
                    result[i] = ' '
                i += 1
            continue

        if string_delimiter is not None:
            if string_delimiter == '`':
                if c == '`':
                    string_delimiter = None
            elif string_delimiter == '"':
                if c == '"' and not is_escaped(code, i):
                    string_delimiter = None
            elif string_delimiter == "'":
                if c == "'" and not is_escaped(code, i):
                    string_delimiter = None
            i += 1
            continue

        if in_verbatim_string:
            if c == '"' and next_c == '"':
                i += 2
            else:
                if c == '"' and next_c != '"':
                    in_verbatim_string = False
                i += 1
            continue

        if c == '@' and next_c == '"':
            in_verbatim_string = True
            i += 2
            continue

        if c == "'" or c == '"' or c == '`':
            string_delimiter = c
            i += 1
            continue

        if c == '/' and next_c == '/':
            result[i] = result[i + 1] = ' '
            in_single_line_comment = True
            i += 2
            continue

        if c == '/' and next_c == '*':
            result[i] = result[i + 1] = ' '
            in_multi_line_comment = True
            i += 2
            continue

        i += 1

    return ''.join(result)


def strip_php_comments(code: str) -> str:
    result = list(code)
    i = 0
    n = len(code)

    in_single_line_comment = False
    in_multi_line_comment = False
    string_delimiter = None
    in_heredoc = False
    heredoc_id = None

    while i < n:
        c = code[i]
        next_c = code[i + 1] if i + 1 < n else ''

        if i == 0 or code[i - 1] == '\n':
            line_start = i
        else:
            line_start = None

        if in_heredoc:
            if line_start is not None:
                j = line_start
                k = 0
                while j < n and k < len(heredoc_id) and code[j] == heredoc_id[k]:
                    j += 1
                    k += 1
                if k == len(heredoc_id) and (j == n or code[j] in (';', '\n')):
                    in_heredoc = False
                    heredoc_id = None
            i += 1
            continue

        if in_single_line_comment:
            if c != '\n':
                result[i] = ' '
            else:
                in_single_line_comment = False
            i += 1
            continue

        if in_multi_line_comment:
            if c == '*' and next_c == '/':
                result[i] = result[i + 1] = ' '
                in_multi_line_comment = False
                i += 2
            else:
                if c != '\n':
                    result[i] = ' '
                i += 1
            continue

        if string_delimiter is not None:
            if c == string_delimiter and not is_escaped(code, i):
                string_delimiter = None
            i += 1
            continue

        if c == '<' and code[i:i+3] == '<<<':
            j = i + 3

            while j < n and code[j].isspace():
                j += 1

            if j < n and code[j] in ("'", '"'):
                quote = code[j]
                j += 1
                start = j
                while j < n and code[j] != quote:
                    j += 1
                heredoc_id = code[start:j]
                j += 1
            else:
                start = j
                while j < n and (code[j].isalnum() or code[j] == '_'):
                    j += 1
                heredoc_id = code[start:j]

            in_heredoc = True
            i = j
            continue


        if c == "'" or c == '"':
            string_delimiter = c
            i += 1
            continue

        if c == '/' and next_c == '/':
            result[i] = result[i + 1] = ' '
            in_single_line_comment = True
            i += 2
            continue

        if c == '#':
            result[i] = ' '
            in_single_line_comment = True
            i += 1
            continue

        if c == '/' and next_c == '*':
            result[i] = result[i + 1] = ' '
            in_multi_line_comment = True
            i += 2
            continue

        i += 1

    return ''.join(result)


def strip_python_comments(code: str) -> str:
    result = list(code)
    i = 0
    n = len(code)

    string_delimiter = None
    in_comment = False

    while i < n:
        c = code[i]

        if in_comment:
            if c != '\n':
                result[i] = ' '
            else:
                in_comment = False
            i += 1
            continue

        if string_delimiter is not None:
            if c == string_delimiter and not is_escaped(code, i):
                string_delimiter = None
            i += 1
            continue

        if c in ("'", '"'):
            string_delimiter = c
            i += 1
            continue

        if c == '#':
            result[i] = ' '
            in_comment = True
            i += 1
            continue

        i += 1

    return ''.join(result)


def remove_python_docstrings(code: str) -> str:
    tokens = tokenize.generate_tokens(io.StringIO(code).readline)
    result = []

    scope_stack = [True]

    for tok in tokens:
        tok_type, tok_str, _, _, _ = tok

        if tok_type == tokenize.INDENT:
            scope_stack.append(True)
        elif tok_type == tokenize.DEDENT:
            scope_stack.pop()
        elif tok_type == tokenize.STRING and scope_stack[-1]:
            scope_stack[-1] = False
            continue
        elif tok_type not in (tokenize.NL, tokenize.NEWLINE):
            scope_stack[-1] = False

        result.append(tok)

    return tokenize.untokenize(result)


def strip_python_comments_and_docstrings(code: str) -> str:
    code = strip_python_comments(code)
    code = remove_python_docstrings(code)
    return code


def is_escaped(result, i):
    count = 0
    i -= 1
    while i >= 0 and result[i] == '\\':
        count += 1
        i -= 1
    return count % 2 == 1


def normalize_whitespace(code: str) -> str:
    lines = code.splitlines()
    normalized = []

    for line in lines:
        stripped = line.rstrip()

        if stripped:
            m = re.match(r'^(\s*)(.*)$', stripped)
            indent, content = m.groups()
            content = re.sub(r' {2,}', ' ', content)
            normalized.append(indent + content)

    return '\n'.join(normalized)


def remove_jcg_comments(code: str) -> str:
    return normalize_whitespace(strip_jcg_comments(code))


def remove_php_comments(code: str) -> str:
    return normalize_whitespace(strip_php_comments(code))


def remove_python_comments(code: str) -> str:
    return normalize_whitespace(strip_python_comments_and_docstrings(code))


examples_cpp = [
    "int x = 5; // simple comment",
    "/* block comment */ \nint x = 5;",
    "int x = /* comment */ 5;",
    'std::string s = "this is not // a comment";',
    'std::string s = "not a /* comment */ either";',
    'std::string s = "escaped quote: \" // still string";',
    "char c1 = '/';\nchar c2 = '*';\nchar c3 = '\'';",
    """std::string s = "/* not a comment */";
int x = 5; /* real comment */
std::string t = "// also not a comment";""",
    """int x = 5;
/*
multi
line
comment
*/
int y = 6;"""
]

examples_c  = [
    """int x = 5; // this is a comment
int y = 6;
""",
    """/* this is a block comment */
int x = 10;
""",
    "int x = /* inline */ 42;",
    """char *s = "this is not // a comment";
char *t = "neither is /* this */";
""",
    'char *s = "escaped quote: \" // still string;',
    """int a = 1;
/*
multi
line
comment
*/
int b = 2;
"""
]

examples_cs = [
    """int x = 5; // comment
int y = 6;
""",
    """/* block comment */
int x = 10;
""",
    """/// <summary>
/// Adds two numbers
/// </summary>
int Add(int a, int b) {
    a = 5;
    return a + b;
}
""",
    'string s = "this is not // a comment";',
    """string s = @"this is not // a comment
and neither is /* this */";
""",
    """string s = "escaped \" // still string";
string t = @"verbatim "" // still string";
/* real comment */
int x = 5;
"""
]

examples_java = [
    """/**
 * Adds two numbers
 */
int add(int a, int b) {
    return a + b; // simple add
}
""",
    'String s = "this is not // a comment";'
]

examples_js = [
    """const s = "not // a comment";
const t = 'not /* this */ either';
""",
    """const s = `this is not // a comment
and not /* this */ either`;
""",
    """const s = "escaped \" // still string";
/* real comment */
const t = `template // still string`;
"""
]

examples_go = [
    """func add(a int, b int) int {
    return a + b // simple add
}
""",
    """/*
multi
line
comment
*/
var x = 10
""",
    's := "this is not // a comment"',
    """s := `this is not // a comment
and not /* this */ either`
""",
    """s := "escaped \" // still string"
/* real comment */
t := `raw // still string`
"""
]

examples_php = [
    """$sql = <<<SQL
SELECT * FROM users
-- not a comment
/* not a comment */
SQL;
""",
    """$txt = <<<'TXT'
This is not // a comment
Nor /* this */
TXT;
""",
    """$sql = <<<SQL
SELECT * FROM users
SQL;
// real comment
echo "done";
""",
    """<?php
$x = 5; // comment
$y = 6;
""",
    "$value = 10; # another comment",
    """/*
multi
line
comment
*/
echo "hello";
""",
    """echo "this is not // a comment";
echo 'nor is /* this */';
""",
    """echo "escaped \" // still string";
echo 'escaped \' # still string';
""",
    """function add($a, $b) {
    return $a + $b; // simple add
}
"""
]

examples_python = [
    '''"""This is a module docstring"""
x = 1
''',
    '''def f():
    """Function docstring"""
    return 1
''',
    '''def f():
    x = """not a docstring"""
    return x
''',
    '''def f():
    # comment
    """docstring"""
    return 1  # end
''',
    '''class A:
    """class docstring"""
    def f(self):
        """method docstring"""
        x = """data"""
        return x
'''
]

# for example in examples_go:
#     print("Initial code sequence:")
#     print(example)
#     print("Code without comments:")
#     print(remove_jcg_comments(example))
#     print("")
#
# with open("comment.txt", "r", encoding="utf-8") as f:
#     code = f.read()
#     print("Initial code sequence:")
#     print(code)
#     print("Code without comments:")
#     print(remove_jcg_comments(code))

for example in examples_php:
    print("Initial code sequence:")
    print(example)
    print("Code without comments:")
    print(remove_php_comments(example))
    print("")

# for example in examples_python:
#     print("Initial code sequence:")
#     print(example)
#     print("Code without comments:")
#     print(remove_python_comments(example))
#     print("")
