'use strict';
const { filter } = hexo.extend;
const cheerio = require('cheerio');

// helper: grab inline background‑image from #page-header
function getBg(header) {
  const style = header.attr('style') || '';
  const m = /background-image:\s*([^;]+)/i.exec(style);
  return m ? m[1] : null;
}

function insertTopImg($) {
  const header = $('#page-header');
  if (header.length === 0) return;

  const bg = getBg(header);
  if (!bg) return;                         // nothing to copy

  // work for posts, pages, archives, tags, categories
  $('#post, #page, #archive, #tag, #category, .page').prepend(
    `<div class="top-img" style="background-image:${bg};"></div>`
  );
}

filter.register('after_render:html', str => {
  const $ = cheerio.load(str, { decodeEntities: false });
  insertTopImg($);
  return $.html();
});
