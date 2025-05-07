'use strict';
const { filter } = hexo.extend;
const cheerio = require('cheerio');

const ROOT_PATH = hexo.config.root || '/whatever/';

/**
 * Fixes image paths in rendered HTML pages.
 * Ensures that image paths have the correct root prefix.
 * 
 * @param {cheerio.Root} $ - The parsed HTML root element.
 */
function fixImagePaths($) {
    // Fix top image in article pages (banner)
    $('header[style*="background-image"]').each((_, elem) => {
        const header = $(elem);
        let style = header.attr('style');
        if (style && style.includes('background-image:url(/img/')) {
            // Replace /img/ with /whatever/img/
            style = style.replace('background-image:url(/img/', `background-image:url(${ROOT_PATH}img/`);
            header.attr('style', style);
        }
    });

    // Fix cover images in homepage and archive pages
    $('img[src^="img/"]').each((_, elem) => {
        const img = $(elem);
        let src = img.attr('src');
        if (src && !src.startsWith(ROOT_PATH)) {
            // Prepend root path to img/xxx.jpg
            src = ROOT_PATH + src;
            img.attr('src', src);
        }
    });
}

filter.register('after_render:html', (str) => {
    const $ = cheerio.load(str, { decodeEntities: false });
    fixImagePaths($);
    return $.html();
});
