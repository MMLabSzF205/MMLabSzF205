## everblog-adaptor-hexo

Hexo adaptor for [everblog](https://github.com/everblogjs/everblog).

### How to use

```sh
$ cd your_hexo_blog_dir
$ npm i everblog-adaptor-hexo --save
$ vim index.js, add `module.exports = require('everblog-adaptor-hexo')`
Open evernote, create `_config.yml`(see below) and some notes

    title: NSWBMW's blog
    subtitle: lalala
    description: my blog
    author: nswbmw

$ everblog build
$ hexo server
$ open http://localhost:4000/
```
