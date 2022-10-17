---
title: Ajax
chrome:
    format: "A4"
    headerTemplate: '<div></div>'
    footerTemplate: '<div style="width:100%; text-align:center; border-top: 1pt solid #eeeeee; margin: 10px 10px 10px; font-size: 8pt;"> 
    <span class=pageNumber></span> / <span class=totalPages></span></div>'
    displayHeaderFooter: true
    margin:
        top: '40px'
        bottom: '65px'
        left: '40px'
        right: '40px'
---

<h1>Ajax</h1>

<!-- @import "[TOC]" {cmd="toc" depthFrom=1 depthTo=6 orderedList=false} -->

<!-- code_chunk_output -->

- [1 Ajax介绍](#1-ajax介绍)
  - [1.1 Ajax简介](#11-ajax简介)
  - [1.2 XML简介](#12-xml简介)
  - [1.3 HTTP协议简介](#13-http协议简介)
- [2 原生Ajax请求](#2-原生ajax请求)

<!-- /code_chunk_output -->

# 1 Ajax介绍

## 1.1 Ajax简介

Ajax 全称为 Asynchronous JacaScript And XML（异步 JavaScript 与 XML）。

通过 Ajax 可以在浏览器中向服务器发送异步请求，无刷新地获取数据。

Ajax 不是新的编程语言，而是一种将现有的标准组合在一起使用的新方式。

Ajax 的优点：

1. 可以无需刷新页面与服务器端进行通信。
2. 允许根据用户事件来更新部分页面内容。

Ajax 的缺点：

1. 没有浏览历史，不能回退。
2. 存在跨域问题。
3. SEO（搜索引擎优化）不友好。

## 1.2 XML简介

XML 全称为**可扩展标记语言**，用于传输和存储数据。

XML 和 HTML 类似，也是由标签组成。不同的是 HTML 中都是预定义标签；而 XML 中没有预定义标签，都是自定义标签，用来表示数据。

## 1.3 HTTP协议简介

HTTP 协议全称为 Hypertext Transport Protocol（超文本传输协议），规定了浏览器和万维网服务器之间相互通信的规则。

请求报文包括 4 部分：请求行、请求头、空行、请求体。例如：

<div align="center">
    <img src="https://cdn.jsdelivr.net/gh/zzx-JLU/images_for_markdown@main/Ajax/HTTP请求报文.161eok7vp18g.png">
</div>

响应报文也包括 4 部分：响应行、响应头、空行、响应体。例如：

<div align="center">
    <img src="https://cdn.jsdelivr.net/gh/zzx-JLU/images_for_markdown@main/Ajax/HTTP响应报文.63le8p6izr00.png">
</div>

# 2 原生Ajax请求

例如：

```html
<!DOCTYPE html>
<html>
    <head>
        <meta http-equiv="pragma" content="no-cache">
        <meta http-equiv="cache-control" content="no-cache">
        <meta http-equiv="Expires" content="0">
        <meta http-equiv="Content-Type" content="text/html; charset=UTF-8">
        <title>title</title>
        <script>
            // 发起 Ajax 请求，访问 AjaxServlet 中的 javaScriptAjax() 方法
            function ajaxRequest()
            {
                // 1. 创建 XMLHttpRequest 对象
                var xmlhttprequest = new XMLHttpRequest();

                // 2. 调用 open(method, url, async) 方法设置请求参数
                //    method：请求的类型，取值为 GET 或 POST
                //    url：文件在服务器上的位置
                //    async：true 表示异步，false 表示同步
                xmlhttprequest.open("GET",
                                    "http://localhost:8080/projectName/servletPath?key=value",
                                    true);
                
                // 3. 绑定 onreadystatechange 事件，处理请求完成后的操作
                xmlhttprequest.onreadystatechange = function() {
                    if (xmlhttprequest.readyState === 4 && xmlhttprequest.status === 200)
                    {
                        document.getElementById("div01").innerHTML =
                            xmlhttprequest.responseText;
                    }
                }
                
                // 4. 调用 send() 方法发送请求
                xmlhttprequest.send();
            }
        </script>
    </head>
    <body>
        <button onclick="ajaxRequest()">ajax request</button>
        <div id="div01"></div>
    </body>
</html>
```
