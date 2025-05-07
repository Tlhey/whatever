'use strict';
const { filter } = hexo.extend;
const cheerio = require('cheerio');

/**
 * Insert the top image on all pages
 * @param {cheerio.Root} $ Root element of the page
 */
function insertTopImg($) {
    const header = $('#page-header');
    if (header.length === 0) return;

    const background = header.css('background-image');
    if (!background) return;

    // Selectors covering home, pagination, post, archive, tag, category, and custom pages
    $('#post, #page, #archive, #tag, #category, .page').prepend(
        `<div class="top-img" style="background-image: ${background};"></div>`
    );
}

// Modify HTML after rendering
filter.register('after_render:html', (str, data) => {
    const $ = cheerio.load(str, {
        decodeEntities: false
    });
    insertTopImg($);
    return $.html();
});
