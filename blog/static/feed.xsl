<?xml version="1.0" encoding="utf-8"?>
<!--
  Human-friendly rendering for the RSS feed. Browsers show raw XML for feeds;
  this XSLT turns the same document into a readable page with subscribe
  instructions, while feed readers keep consuming the underlying XML.
-->
<xsl:stylesheet version="1.0"
  xmlns:xsl="http://www.w3.org/1999/XSL/Transform"
  xmlns:atom="http://www.w3.org/2005/Atom">
  <xsl:output method="html" version="1.0" encoding="utf-8" indent="yes"
    doctype-system="about:legacy-compat"/>

  <xsl:template match="/rss/channel">
    <html lang="en">
      <head>
        <meta charset="utf-8"/>
        <meta name="viewport" content="width=device-width, initial-scale=1"/>
        <meta name="robots" content="noindex"/>
        <title><xsl:value-of select="title"/> &#8226; RSS feed</title>
        <style>
          :root { color-scheme: light dark; }
          body {
            font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, Helvetica, Arial, sans-serif;
            line-height: 1.6; margin: 0; padding: 0 1rem;
            color: #1a1a1a; background: #fafafa;
          }
          .wrap { max-width: 44rem; margin: 0 auto; padding: 2.5rem 0 4rem; }
          .banner {
            border: 1px solid #d9c86b; background: #fff8dc; color: #665c1e;
            border-radius: 10px; padding: 1rem 1.25rem; margin: 0 0 2rem;
            font-size: 0.95rem;
          }
          .banner strong { color: #4a4212; }
          .banner code {
            background: rgba(0,0,0,.07); padding: .1em .4em; border-radius: 5px;
            word-break: break-all;
          }
          h1 { font-size: 1.9rem; margin: 0 0 .25rem; }
          .desc { color: #555; margin: 0 0 2rem; }
          .item { padding: 1.25rem 0; border-top: 1px solid #e5e5e5; }
          .item h2 { font-size: 1.2rem; margin: 0 0 .35rem; }
          .item h2 a { color: #0b5cad; text-decoration: none; }
          .item h2 a:hover { text-decoration: underline; }
          .item time { color: #888; font-size: .85rem; }
          .item .summary { margin: .5rem 0 0; color: #333; }
          a { color: #0b5cad; }
          @media (prefers-color-scheme: dark) {
            body { color: #e6e6e6; background: #16181c; }
            .desc { color: #aaa; }
            .item { border-top-color: #2a2d33; }
            .item .summary { color: #cfcfcf; }
            .item time { color: #888; }
            .banner { border-color: #6a5f1e; background: #26240f; color: #e8dfa0; }
            .banner strong { color: #f2ead0; }
            .item h2 a, a { color: #6cb6ff; }
          }
        </style>
      </head>
      <body>
        <div class="wrap">
          <div class="banner">
            <strong>This is an RSS feed.</strong>
            Subscribe by copying the address from your browser's location bar into
            your feed reader (e.g. Feedly, Inoreader, NetNewsWire), or click the
            RSS icon in an app that supports feeds. You'll get new posts
            automatically &#8212; no email required.
          </div>

          <h1><xsl:value-of select="title"/></h1>
          <p class="desc"><xsl:value-of select="description"/></p>

          <xsl:for-each select="item">
            <div class="item">
              <h2>
                <a href="{link}"><xsl:value-of select="title"/></a>
              </h2>
              <time><xsl:value-of select="pubDate"/></time>
              <div class="summary"><xsl:value-of select="description" disable-output-escaping="yes"/></div>
            </div>
          </xsl:for-each>
        </div>
      </body>
    </html>
  </xsl:template>
</xsl:stylesheet>
