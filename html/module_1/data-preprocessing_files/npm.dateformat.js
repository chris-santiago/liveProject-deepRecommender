"use strict";(self.webpackChunkliveVideoAndProject=self.webpackChunkliveVideoAndProject||[]).push([[106],{1280:(t,e,n)=>{n.d(e,{ZP:()=>o});var r=/d{1,4}|D{3,4}|m{1,4}|yy(?:yy)?|([HhMsTt])\1?|W{1,2}|[LlopSZN]|"[^"]*"|'[^']*'/g,a=/\b(?:[A-Z]{1,3}[A-Z][TC])(?:[-+]\d{4})?|((?:Australian )?(?:Pacific|Mountain|Central|Eastern|Atlantic) (?:Standard|Daylight|Prevailing) Time)\b/g,u=/[^-+\dA-Z]/g;function o(t,e,n,a){if(1!==arguments.length||"string"!=typeof t||/\d/.test(t)||(e=t,t=void 0),(t=t||0===t?t:new Date)instanceof Date||(t=new Date(t)),isNaN(t))throw TypeError("Invalid date");var u=(e=String(i[e]||e||i.default)).slice(0,4);"UTC:"!==u&&"GMT:"!==u||(e=e.slice(4),n=!0,"GMT:"===u&&(a=!0));var o=function(){return n?"getUTC":"get"},l=function(){return t[o()+"Date"]()},M=function(){return t[o()+"Day"]()},D=function(){return t[o()+"Month"]()},h=function(){return t[o()+"FullYear"]()},T=function(){return t[o()+"Hours"]()},g=function(){return t[o()+"Minutes"]()},N=function(){return t[o()+"Seconds"]()},p=function(){return t[o()+"Milliseconds"]()},v=function(){return n?0:t.getTimezoneOffset()},H=function(){return y(t)},S=function(){return c(t)},b={d:function(){return l()},dd:function(){return d(l())},ddd:function(){return m.dayNames[M()]},DDD:function(){return s({y:h(),m:D(),d:l(),_:o(),dayName:m.dayNames[M()],short:!0})},dddd:function(){return m.dayNames[M()+7]},DDDD:function(){return s({y:h(),m:D(),d:l(),_:o(),dayName:m.dayNames[M()+7]})},m:function(){return D()+1},mm:function(){return d(D()+1)},mmm:function(){return m.monthNames[D()]},mmmm:function(){return m.monthNames[D()+12]},yy:function(){return String(h()).slice(2)},yyyy:function(){return d(h(),4)},h:function(){return T()%12||12},hh:function(){return d(T()%12||12)},H:function(){return T()},HH:function(){return d(T())},M:function(){return g()},MM:function(){return d(g())},s:function(){return N()},ss:function(){return d(N())},l:function(){return d(p(),3)},L:function(){return d(Math.floor(p()/10))},t:function(){return T()<12?m.timeNames[0]:m.timeNames[1]},tt:function(){return T()<12?m.timeNames[2]:m.timeNames[3]},T:function(){return T()<12?m.timeNames[4]:m.timeNames[5]},TT:function(){return T()<12?m.timeNames[6]:m.timeNames[7]},Z:function(){return a?"GMT":n?"UTC":f(t)},o:function(){return(v()>0?"-":"+")+d(100*Math.floor(Math.abs(v())/60)+Math.abs(v())%60,4)},p:function(){return(v()>0?"-":"+")+d(Math.floor(Math.abs(v())/60),2)+":"+d(Math.floor(Math.abs(v())%60),2)},S:function(){return["th","st","nd","rd"][l()%10>3?0:(l()%100-l()%10!=10)*l()%10]},W:function(){return H()},WW:function(){return d(H())},N:function(){return S()}};return e.replace(r,(function(t){return t in b?b[t]():t.slice(1,t.length-1)}))}var i={default:"ddd mmm dd yyyy HH:MM:ss",shortDate:"m/d/yy",paddedShortDate:"mm/dd/yyyy",mediumDate:"mmm d, yyyy",longDate:"mmmm d, yyyy",fullDate:"dddd, mmmm d, yyyy",shortTime:"h:MM TT",mediumTime:"h:MM:ss TT",longTime:"h:MM:ss TT Z",isoDate:"yyyy-mm-dd",isoTime:"HH:MM:ss",isoDateTime:"yyyy-mm-dd'T'HH:MM:sso",isoUtcDateTime:"UTC:yyyy-mm-dd'T'HH:MM:ss'Z'",expiresHeaderFormat:"ddd, dd mmm yyyy HH:MM:ss Z"},m={dayNames:["Sun","Mon","Tue","Wed","Thu","Fri","Sat","Sunday","Monday","Tuesday","Wednesday","Thursday","Friday","Saturday"],monthNames:["Jan","Feb","Mar","Apr","May","Jun","Jul","Aug","Sep","Oct","Nov","Dec","January","February","March","April","May","June","July","August","September","October","November","December"],timeNames:["a","p","am","pm","A","P","AM","PM"]},d=function(t){var e=arguments.length>1&&void 0!==arguments[1]?arguments[1]:2;return String(t).padStart(e,"0")},s=function(t){var e=t.y,n=t.m,r=t.d,a=t._,u=t.dayName,o=t.short,i=void 0!==o&&o,m=new Date,d=new Date;d.setDate(d[a+"Date"]()-1);var s=new Date;return s.setDate(s[a+"Date"]()+1),m[a+"FullYear"]()===e&&m[a+"Month"]()===n&&m[a+"Date"]()===r?i?"Tdy":"Today":d[a+"FullYear"]()===e&&d[a+"Month"]()===n&&d[a+"Date"]()===r?i?"Ysd":"Yesterday":s[a+"FullYear"]()===e&&s[a+"Month"]()===n&&s[a+"Date"]()===r?i?"Tmw":"Tomorrow":u},y=function(t){var e=new Date(t.getFullYear(),t.getMonth(),t.getDate());e.setDate(e.getDate()-(e.getDay()+6)%7+3);var n=new Date(e.getFullYear(),0,4);n.setDate(n.getDate()-(n.getDay()+6)%7+3);var r=e.getTimezoneOffset()-n.getTimezoneOffset();e.setHours(e.getHours()-r);var a=(e-n)/6048e5;return 1+Math.floor(a)},c=function(t){var e=t.getDay();return 0===e&&(e=7),e},f=function(t){return(String(t).match(a)||[""]).pop().replace(u,"").replace(/GMT\+0000/g,"UTC")}}}]);